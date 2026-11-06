"""Separate pretraining, validation selection and complete epoch-boundary resume."""
import copy
import inspect
import os
import random
from pathlib import Path

import numpy as np
import torch

from opencood.tools.codealign_protocol import require, sha256, stage_parents


def _parent_identity(parents):
    """Paths and old metadata are informational; actual weight dependencies matter."""
    require(isinstance(parents, dict), "invalid checkpoint parents")
    result = {}
    for stage, record in parents.items():
        require(isinstance(record, dict) and record.get("sha256") and "epoch" in record,
                f"incomplete parent checkpoint identity: {stage}")
        result[stage] = {"sha256": record["sha256"], "epoch": record["epoch"]}
    return result


def verify_checkpoint_identity(saved, expected, config=None, experiment=None):
    """Check dataset, stage and actual parent weights, including old checkpoints."""
    require(isinstance(saved, dict) and isinstance(expected, dict), "missing checkpoint identity")
    require(saved.get("dataset") == expected.get("dataset") and
            saved.get("stage") == expected.get("stage"), "checkpoint dataset/stage drift")
    require(_parent_identity(saved.get("parents", {})) == _parent_identity(expected.get("parents", {})),
            "checkpoint parent weights changed")


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
    require(isinstance(value, dict) and value.get("format") == "codealign_protocol_v2", f"unrecognized checkpoint: {path}")
    return value


def provenance(config, parents=None):
    return {"dataset": config["codealign_protocol"]["dataset"],
            "stage": config["codealign_protocol"]["stage"], "parents": parents or {}}


def parent_checkpoints(experiment, config):
    from opencood.tools.codealign_protocol import load_config
    import json

    experiment = Path(experiment)
    dataset = config["codealign_protocol"]["dataset"]
    checked = {}

    def completed(stage):
        if stage in checked:
            return checked[stage]
        directory = experiment / stage
        path = directory / "best.pth"
        completion = directory / "complete.json"
        require(completion.is_file(), f"parent stage is not complete: {stage}")
        complete = json.loads(completion.read_text(encoding="utf-8"))
        require(path.is_file(), f"missing parent checkpoint: {stage}")
        checksum = sha256(path)
        require(complete["checkpoint_sha256"] == checksum, f"parent checkpoint changed: {stage}")
        parent_config = load_config(directory / "config.yaml")
        require(parent_config["codealign_protocol"]["stage"] == stage and
                parent_config["codealign_protocol"]["dataset"] == dataset, "cross-dataset/stage parent")
        # Verify dependency records without retaining every ancestor model and
        # optimizer in memory. Only direct parents supply tensors for assembly.
        ancestors = {parent: completed(parent)[0] for parent in stage_parents(stage)}
        expected = provenance(parent_config, ancestors)
        verify_checkpoint_identity(complete["provenance"], expected)
        require(isinstance(complete.get("best_epoch"), int), f"invalid parent epoch: {stage}")
        record = {"path": str(path.resolve()), "sha256": checksum, "epoch": complete["best_epoch"]}
        checked[stage] = record, expected
        return checked[stage]

    states, records = {}, {}
    for stage in stage_parents(config["codealign_protocol"]["stage"]):
        record, expected = completed(stage)
        value = load_checkpoint(record["path"])
        verify_checkpoint_identity(value["provenance"], expected)
        require(value.get("kind") == "training" and value.get("epoch") == record["epoch"],
                f"invalid parent training checkpoint: {stage}")
        states[stage] = value["model"]
        records[stage] = record
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


def validate_model_state(model, saved_model):
    """Prevent strict load_state_dict from silently casting checkpoint tensors."""
    current = model.state_dict()
    require(isinstance(saved_model, dict) and set(current) == set(saved_model), "resume model keys differ")
    for key, tensor in current.items():
        saved = saved_model[key]
        require(isinstance(saved, torch.Tensor) and saved.shape == tensor.shape and saved.dtype == tensor.dtype,
                f"resume model tensor incompatible: {key}")


def restore_training(path, model, optimizer, scheduler, generators, expected_provenance, config=None, experiment=None):
    state = load_checkpoint(path)
    old_config = state.get("config")
    verify_checkpoint_identity(state["provenance"], expected_provenance)
    require(state["kind"] == "training", "final assembly is not a training resume")
    validate_model_state(model, state["model"])
    if config is not None:
        require(isinstance(state.get("epoch"), int) and state["epoch"] >= 0,
                "invalid completed epoch in checkpoint")
        require(config["train_params"]["epoches"] > state["epoch"], "total epochs must exceed completed epochs")
    model.load_state_dict(state["model"], strict=True)
    unknown_settings = config is not None and old_config is None
    optimizer_changed = config is not None and (unknown_settings or
        old_config["optimizer"]["core_method"] != config["optimizer"]["core_method"] or
        old_config["optimizer"].get("args", {}) != config["optimizer"].get("args", {}))
    scheduler_changed = config is not None and (unknown_settings or
        old_config["lr_scheduler"] != config["lr_scheduler"])
    rate_changed = config is not None and (unknown_settings or
        old_config["optimizer"]["lr"] != config["optimizer"]["lr"])
    # These schedulers can retain progress with an explicitly rebased next-step
    # rate. Others (e.g. warmup, OneCycle and Cyclic) derive rates/momentum from
    # additional configured bounds, so a rate change restarts their new schedule.
    rebase_supported = type(scheduler).__name__ in {
        "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"}
    restart_scheduler = optimizer_changed or scheduler_changed or (rate_changed and not rebase_supported)
    configured_groups = [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
                         for group in optimizer.param_groups]
    fresh_scheduler = copy.deepcopy(scheduler.state_dict())
    policy = {}
    if not optimizer_changed:
        optimizer.load_state_dict(state["optimizer"])
        policy["optimizer"] = "restored compatible state and arguments"
    else:
        # Arguments such as Adam.amsgrad or RMSprop.centered change state
        # structure. Reusing their old moments can fail only on the next step.
        optimizer.state.clear()
        policy["optimizer"] = "reinitialized: type/arguments changed or old settings unavailable"
    if not restart_scheduler:
        scheduler.load_state_dict(state["scheduler"])
        policy["scheduler"] = "restored compatible progress"
        if rate_changed:
            lr = float(config["optimizer"]["lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["initial_lr"] = lr
            scheduler.base_lrs = [lr] * len(optimizer.param_groups)
            scheduler._last_lr = [lr] * len(optimizer.param_groups)
            policy["learning_rate"] = "current YAML rate applied to next optimizer step"
        else:
            policy["learning_rate"] = "retained compatible scheduled rate"
    else:
        # Preserve all fresh scheduler-derived group settings, including warmup
        # LR and OneCycle/Cyclic bounds and momentum. A nominal optimizer LR
        # must not replace a schedule's explicitly configured starting rate.
        for group, configured in zip(optimizer.param_groups, configured_groups):
            parameters = group["params"]
            group.clear()
            group.update(configured, params=parameters)
        scheduler.load_state_dict(fresh_scheduler)
        policy["scheduler"] = "reinitialized from current YAML; schedule restarts at initial step"
        policy["learning_rate"] = "current YAML scheduler initialization determines effective rate"
    if config is not None:
        policy["configured_optimizer_lr"] = float(config["optimizer"]["lr"])
    policy["effective_learning_rates"] = [float(group["lr"]) for group in optimizer.param_groups]
    if config is None or (old_config is not None and old_config.get("seed") == config["seed"]):
        restore_rng(state["rng"], generators)
        policy["rng"] = "restored"
    else:
        seed_everything(config["seed"], config.get("reproducibility"))
        for offset, (name, generator) in enumerate(generators.items()):
            generator.manual_seed(config["seed"] + {"train": 0, "validate": 1}.get(name, offset))
        policy["rng"] = "reseeded from current YAML: seed changed or old seed unavailable"
    state["restore_policy"] = policy
    print("Resume policy: " + "; ".join(f"{key}={value}" for key, value in policy.items()))
    return state
