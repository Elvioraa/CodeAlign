"""Separate pretraining, validation selection and complete epoch-boundary resume."""
import hashlib
import inspect
import os
import random
from pathlib import Path

import numpy as np
import torch

from opencood.tools.codealign_protocol import ROOT, digest, require, sha256, stage_parents


def code_fingerprint():
    return digest({p.relative_to(ROOT).as_posix(): sha256(p)
                   for p in sorted((ROOT / "opencood").rglob("*.py"))})


def seed_everything(seed, options=None):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    options = options or {}
    torch.backends.cudnn.benchmark = options.get("cudnn_benchmark", False)
    torch.backends.cudnn.deterministic = options.get("cudnn_deterministic", True)
    torch.use_deterministic_algorithms(options.get("deterministic_algorithms", False))


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def rng_state(generators):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {k: v.get_state() for k, v in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        require(torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count(),
                "CUDA device topology changed on resume")
        torch.cuda.set_rng_state_all(state["cuda"])
    require(set(state["generators"]) == set(generators), "loader generator identity mismatch")
    for k, value in state["generators"].items():
        generators[k].set_state(value)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    path = Path(path)
    require(path.is_file(), f"missing checkpoint (never fall back to random weights): {path}")
    # Trusted local training files include Python/NumPy RNG state. Never load
    # untrusted downloaded pickle checkpoints through this full-state reader.
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    value = torch.load(path, **options)
    require(value.get("format") == "codealign_protocol_v2", f"unrecognized checkpoint: {path}")
    return value


def provenance(config, lock_hash, parents=None):
    return {"dataset": config["codealign_protocol"]["dataset"], "stage": config["codealign_protocol"]["stage"],
            "config_sha256": digest(config), "lock_sha256": lock_hash,
            "code_sha256": code_fingerprint(), "parents": parents or {}}


def parent_checkpoints(experiment, config, lock_hash):
    from opencood.tools.codealign_protocol import load_config, read_yaml
    experiment = Path(experiment)
    states, records = {}, {}
    for stage in stage_parents(config["codealign_protocol"]["stage"]):
        directory = experiment / stage
        path = directory / "best.pth"
        completion = directory / "complete.json"
        require(completion.is_file(), f"parent stage is not complete: {stage}")
        import json
        complete = json.loads(completion.read_text())
        require(complete["checkpoint_sha256"] == sha256(path), f"parent checkpoint changed: {stage}")
        value = load_checkpoint(path)
        parent_config = load_config(directory / "config.yaml")
        expected = provenance(parent_config, lock_hash, value["provenance"]["parents"])
        require(value["provenance"] == expected, f"parent config/code/dataset mismatch: {stage}")
        require(value["provenance"]["dataset"] == config["codealign_protocol"]["dataset"], "cross-dataset parent")
        states[stage] = value["model"]
        records[stage] = {"path": str(path.resolve()), "sha256": sha256(path),
                          "epoch": value["epoch"], "config_sha256": value["provenance"]["config_sha256"]}
    return states, records


def initialize_model(model, stage, states):
    current = model.state_dict()
    owners = {}
    source = "m1" if stage == "final_infer" else stage.split("/")[1].split("_")[0]
    for key in current:
        if stage.startswith("single/"):
            continue
        if stage.startswith("code_space/"):
            if key.startswith(("aligners.", "codebooks.")):
                continue
            parent = f"single/{source}"
        elif stage.startswith("fcf/"):
            if key.startswith("translators."):
                continue
            parent = f"code_space/{source}" if key.startswith(("encoders.", "backbones.", "aligners.")) else "code_space/m1"
        else:
            if key.startswith("translators."):
                parent = f"fcf/{key.split('.')[1]}_to_m1"
            elif key.startswith(("encoders.", "backbones.", "aligners.", "codebooks.")):
                parent = f"code_space/{key.split('.')[1]}"
            else:
                parent = "code_space/m1"
        require(parent in states and key in states[parent], f"missing owned tensor {parent}:{key}")
        tensor = states[parent][key]
        require(tensor.shape == current[key].shape and tensor.dtype == current[key].dtype,
                f"incompatible tensor {parent}:{key}")
        current[key] = tensor
        owners[key] = parent
    model.load_state_dict(current, strict=True)
    if stage == "final_infer":
        for m in ("m2", "m3", "m4"):
            translated = states[f"fcf/{m}_to_m1"]
            for key, value in translated.items():
                if key.startswith("translators."):
                    continue
                require(key in current and torch.equal(value, current[key]),
                        f"frozen weights changed during FCF: {m}:{key}")
    return owners


def restore_training(path, model, optimizer, scheduler, generators, expected_provenance):
    state = load_checkpoint(path)
    require(state["provenance"] == expected_provenance, "resume identity/config/code/parent drift")
    require(state["kind"] == "training", "final assembly is not a training resume")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"], generators)
    return state
