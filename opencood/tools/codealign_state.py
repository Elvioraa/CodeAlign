"""Separate pretraining, validation selection and complete epoch-boundary resume."""
import hashlib
import ast
import inspect
import os
import random
import io
import subprocess
import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from opencood.tools.codealign_protocol import ROOT, config_root, digest, protocol_config, read_yaml, require, sha256, stage_parents


def code_fingerprint():
    return digest({p.relative_to(ROOT).as_posix(): sha256(p)
                   for p in sorted((ROOT / "opencood").rglob("*.py"))})


def data_model_fingerprint():
    return digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                   for part in ("models", "data_utils", "loss", "utils")
                   for p in sorted((ROOT / "opencood" / part).rglob("*.py"))})


def contract_code_fingerprint():
    # Pin selection/assignment policies while allowing runner maintenance.
    from opencood.tools import codealign_protocol as protocol
    policy = {name: ast.dump(ast.parse(inspect.getsource(getattr(protocol, name))), include_attributes=False)
              for name in ("scene_owner", "_resolved_assignments", "evaluation_config", "validate_prefix")}
    return digest({"model_data": data_model_fingerprint(), "policy": policy})


def selection_policy_fingerprint(source):
    wanted = {"scene_owner", "_resolved_assignments", "evaluation_config", "validate_prefix"}
    tree = ast.parse(source)
    functions = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted}
    require(set(functions) == wanted, "incomplete assignment policy")
    return digest(functions)


@lru_cache(maxsize=1)
def legacy_base_hashes():
    """Identify unmodified e63957b Linux/Windows code without trusting current tools."""
    archive = subprocess.check_output(["git", "archive", "e63957b", "opencood"], cwd=str(ROOT))
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()
                 if m.isfile() and m.name.endswith(".py")}
    require(selection_policy_fingerprint(files["opencood/tools/codealign_protocol.py"].decode("utf-8")) ==
            selection_policy_fingerprint((ROOT / "opencood/tools/codealign_protocol.py").read_text(encoding="utf-8")),
            "legacy assignment policy changed")
    results = []
    for windows in (False, True):
        material = {k: (v.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") if windows else v)
                    for k, v in files.items()}
        full = digest({k: hashlib.sha256(v).hexdigest() for k, v in material.items()})
        core = digest({k: hashlib.sha256(v.replace(b"\r\n", b"\n")).hexdigest() for k, v in material.items()
                       if k.split("/")[1] in ("models", "data_utils", "loss", "utils")})
        results.append((full, core))
    return results


def verify_checkpoint_identity(saved, expected, config, experiment):
    if saved == expected:
        return None
    require(saved.get("dataset") == expected["dataset"] and saved.get("stage") == expected["stage"]
            and saved.get("parents") == expected["parents"], "checkpoint dataset/stage/parent drift")
    # Legacy checkpoints saved only full hashes. Recover an exact old config
    # from an available source, and prove both old code and lock provenance.
    candidates = [config, read_yaml(config_root(expected["dataset"]) / expected["stage"] / "config.yaml")]
    old_config = next((item for item in candidates if digest(item) == saved.get("config_sha256")), None)
    require(old_config is not None and protocol_config(old_config) == protocol_config(config),
            "legacy checkpoint config cannot be recovered or protocol changed")
    require(saved.get("lock_sha256") == sha256(Path(experiment) / "protocol_lock.yaml"),
            "legacy lock provenance cannot be verified")
    require(any(saved.get("code_sha256") == full and data_model_fingerprint() == core
                for full, core in legacy_base_hashes()), "legacy code provenance cannot be verified")
    return old_config


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


def create_scheduler(config, optimizer):
    """Construct the explicitly requested scheduler, without silent fallback."""
    spec = dict(config["lr_scheduler"])
    name = spec.pop("core_method")
    name = {"step": "StepLR", "multistep": "MultiStepLR",
            "exponential": "ExponentialLR", "Exponential": "ExponentialLR"}.get(name, name)
    options = dict(spec.pop("args", {}))
    options.update(spec)
    if name == "MultiStepLR" and "step_size" in options:
        require("milestones" not in options, "duplicate scheduler milestones")
        options["milestones"] = options.pop("step_size")
    scheduler = getattr(torch.optim.lr_scheduler, name, None)
    require(scheduler is not None, f"unsupported scheduler: {name}")
    return scheduler(optimizer, **options)


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
            "config_sha256": digest(protocol_config(config)), "lock_sha256": lock_hash,
            "code_sha256": contract_code_fingerprint(), "parents": parents or {}}


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
        verify_checkpoint_identity(value["provenance"], expected, parent_config, experiment)
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


def restore_training(path, model, optimizer, scheduler, generators, expected_provenance, config=None, experiment=None):
    state = load_checkpoint(path)
    old_config = state.get("config")
    if state["provenance"] != expected_provenance:
        require(config is not None and experiment is not None, "resume identity/config/code/parent drift")
        old_config = verify_checkpoint_identity(state["provenance"], expected_provenance, config, experiment)
        state["config"] = old_config
    require(state["kind"] == "training", "final assembly is not a training resume")
    current = model.state_dict()
    require(set(current) == set(state["model"]), "resume model keys differ")
    for key, tensor in current.items():
        saved = state["model"][key]
        require(saved.shape == tensor.shape and saved.dtype == tensor.dtype,
                f"resume model tensor incompatible: {key}")
    model.load_state_dict(state["model"], strict=True)
    if config is not None:
        require(old_config is not None, "legacy checkpoint lacks saved config; cannot verify resume compatibility")
        require(config["train_params"]["epoches"] > state["epoch"], "total epochs must exceed completed epochs")
    optimizer_changed = config is not None and old_config["optimizer"]["core_method"] != config["optimizer"]["core_method"]
    scheduler_changed = config is not None and old_config["lr_scheduler"] != config["lr_scheduler"]
    configured_groups = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
    if not optimizer_changed:
        optimizer.load_state_dict(state["optimizer"])
        if config is not None:
            for group, configured in zip(optimizer.param_groups, configured_groups):
                for key, value in configured.items():
                    if key not in ("lr", "initial_lr"):
                        group[key] = value
    else:
        print("Optimizer type changed: optimizer state reinitialized; model and RNG restored.")
    if not optimizer_changed and not scheduler_changed:
        scheduler.load_state_dict(state["scheduler"])
    else:
        # New scheduler configuration begins at the completed epoch, with
        # configured LR as the next-step rate; old scheduler state is not mixed.
        scheduler.last_epoch = state["epoch"]
        scheduler._step_count = state["epoch"] + 1
        print("Scheduler configuration changed: scheduler reinitialized at completed epoch.")
    if config is not None and (optimizer_changed or scheduler_changed or
                               config["optimizer"]["lr"] != old_config["optimizer"]["lr"]):
        # Interpret the newly configured learning rate as the next-step rate.
        # Rebase scheduler state too, so future scheduler steps retain it.
        lr = float(config["optimizer"]["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["initial_lr"] = lr
        scheduler.base_lrs = [lr] * len(optimizer.param_groups)
        scheduler._last_lr = [lr] * len(optimizer.param_groups)
    restore_rng(state["rng"], generators)
    return state
