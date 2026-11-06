import copy
import json
import tempfile
import types
from unittest.mock import patch
from pathlib import Path

import pytest
import torch
try:
    import torch._dynamo  # Initialize before the smoke test temporarily patches sys.modules.
except ImportError:  # Older supported PyTorch releases have no Dynamo.
    pass

from opencood.tools.codealign_protocol import config_root, load_config, prepare, read_yaml, sha256
from opencood.tools.codealign_state import (atomic_torch_save, create_scheduler, load_checkpoint,
                                             provenance, parent_checkpoints, restore_training, rng_state, verify_checkpoint_identity)
from opencood.tools.run_codealign_doma import make_loader, summarize, train_stage


def config():
    return load_config(config_root("v2xset") / "single/m1/config.yaml")


def test_identity_contains_only_dataset_stage_and_parent_weights():
    original = config()
    changed = copy.deepcopy(original)
    changed["optimizer"] = {"core_method": "SGD", "lr": 0.03, "args": {"momentum": 0.9}}
    changed["lr_scheduler"] = {"core_method": "step", "step_size": 3, "gamma": 0.8}
    changed["train_params"].update(batch_size=7, epoches=88, eval_freq=7, save_freq=9)
    changed["loader"]["num_workers"] = 3
    changed["seed"] = 999
    assert provenance(original) == provenance(changed)
    assert set(provenance(original)) == {"dataset", "stage", "parents"}


def test_cpu_loader_uses_reviewed_batch_size():
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return 8
        def __getitem__(self, index): return index
        def collate_batch_train(self, batch): return batch
    settings = config()
    settings["loader"].update(num_workers=0, pin_memory=False)
    settings["train_params"]["batch_size"] = 3
    loader = make_loader(Dataset(), settings, torch.Generator().manual_seed(303), train=True)
    assert len(next(iter(loader))) == 3


def test_scientific_notation_is_numeric():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text("lr: 1e-4\nweight_decay: 1e-3\n", encoding="utf-8")
        assert read_yaml(path) == {"lr": 0.0001, "weight_decay": 0.001}


def test_summary_uses_assignment_semantics_not_json_format():
    with tempfile.TemporaryDirectory() as directory:
        for n in range(1, 5):
            path = Path(directory) / "final_infer" / f"CAV{n}"
            path.mkdir(parents=True)
            result = {"dataset": "v2xset", "samples": 1, "sample_sha256": "samples", "gt_sha256": "gt",
                      "checkpoint_sha256": "model", "range": [1], "comm_range": 180,
                      "assignment_members": [["scene", [["1", "m1"]]]],
                      "use_cav": n, "ap30": 0.1, "ap50": 0.1, "ap70": 0.1}
            (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (path / "sample_ledger.json").write_text(json.dumps([{"key": ["scene", "0"], "modalities": ["m1", "m2", "m3", "m4"][:n]}]), encoding="utf-8")
        summarize(directory)
        assert (Path(directory) / "final_infer/ap_summary.csv").is_file()


def test_scheduler_parameters_are_applied_without_fallback():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = config()
    cfg["lr_scheduler"] = {"core_method": "CosineAnnealingLR", "args": {"T_max": 7, "eta_min": 0.01}}
    scheduler = create_scheduler(cfg, optimizer)
    assert scheduler.T_max == 7 and scheduler.eta_min == 0.01
    cfg["lr_scheduler"] = {"core_method": "NotAScheduler", "args": {}}
    with pytest.raises(ValueError, match="unsupported scheduler"):
        create_scheduler(cfg, optimizer)


def test_default_stage_runner_cpu_smoke_without_real_data():
    import opencood.tools
    import yaml
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, train): self.train = train
        def __len__(self): return 4
        def __getitem__(self, index): return float(index)
        def reinitialize(self): pass
        def collate_batch_train(self, batch):
            x = torch.tensor(batch).reshape(-1, 1)
            return {"ego": {"x": x, "label_dict": {"y": x}, "record_len": torch.ones(len(batch))}}
    class Model(torch.nn.Linear):
        def __init__(self): super().__init__(1, 1)
        def model_train_init(self): pass
        def forward(self, data): return {"prediction": super().forward(data["x"])}
    class Loss(torch.nn.Module):
        def forward(self, output, target): return ((output["prediction"] - target["y"]) ** 2).mean()
        def logging(self, *args): pass
    class Writer:
        def __init__(self, *args): pass
        def add_scalar(self, *args): pass
        def close(self): pass
    fake_utils = types.ModuleType("opencood.tools.train_utils")
    fake_utils.create_model = lambda *args, **kwargs: Model()
    fake_utils.create_loss = lambda *args: Loss()
    fake_utils.setup_lr_schedular = lambda cfg, opt: torch.optim.lr_scheduler.MultiStepLR(opt, [3], gamma=0.1)
    fake_utils.to_device = lambda batch, device: batch
    dataset_module = types.ModuleType("opencood.data_utils.datasets.codealign_protocol_dataset")
    dataset_module.build_protocol_dataset = lambda hypes, train: Dataset(train)
    dataset_module.sample_keys = lambda dataset: [["train" if dataset.train else "validate", str(i)] for i in range(4)]
    datasets = types.ModuleType("opencood.data_utils.datasets")
    datasets.__path__ = []
    writer_module = types.ModuleType("tensorboardX")
    writer_module.SummaryWriter = Writer
    modules = {"opencood.tools.train_utils": fake_utils, "tensorboardX": writer_module,
               "opencood.data_utils.datasets": datasets,
               "opencood.data_utils.datasets.codealign_protocol_dataset": dataset_module}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepare(root, "v2xset")
        path = root / "single/m1/config.yaml"
        settings = load_config(path)
        settings["train_params"].update(batch_size=2, epoches=1, eval_freq=1, save_freq=1)
        settings["loader"].update(num_workers=0, pin_memory=False)
        path.write_text(yaml.safe_dump(settings), encoding="utf-8")
        with patch.dict("sys.modules", modules), patch.object(opencood.tools, "train_utils", fake_utils, create=True), \
             patch("opencood.tools.run_codealign_doma.device_for_run", return_value=torch.device("cpu")), \
             patch("opencood.tools.run_codealign_doma.population_members", return_value=[["scene", [["1", "m1"]]]]):
            train_stage(root, "single/m1", False)
        assert (root / "single/m1/complete.json").is_file()
        assert (root / "single/m1/latest.pth").is_file()
        assert len(list((root / "single/m1/run_history").glob("*/run.json"))) == 1


@pytest.fixture
def training_checkpoint(tmp_path):
    old = config()
    old["optimizer"] = {"core_method": "Adam", "lr": 0.01, "args": {"weight_decay": 0.001}}
    old["lr_scheduler"] = {"core_method": "multistep", "step_size": [1, 3], "gamma": 0.1}
    old["train_params"]["epoches"] = 4
    model, optimizer, scheduler, generators = runtime_objects(old)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    state = {"format": "codealign_protocol_v2", "kind": "training", "epoch": 1,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "rng": rng_state(generators),
             "provenance": provenance(old), "config": old, "data_sha256": "actual-sample-population"}
    path = tmp_path / "latest.pth"
    atomic_torch_save(path, state)
    return path, state, old


def runtime_objects(settings):
    model = torch.nn.Linear(3, 1)
    spec = settings["optimizer"]
    optimizer = getattr(torch.optim, spec["core_method"])(model.parameters(), lr=spec["lr"], **spec.get("args", {}))
    scheduler = create_scheduler(settings, optimizer)
    generators = {"train": torch.Generator().manual_seed(settings["seed"]),
                  "validate": torch.Generator().manual_seed(settings["seed"] + 1)}
    return model, optimizer, scheduler, generators


@pytest.mark.parametrize("parameter", ["unchanged", "lr", "batch_size", "epoches", "num_workers", "eval_freq", "save_freq", "seed"])
def test_resume_accepts_current_training_settings(training_checkpoint, parameter):
    path, saved, old = training_checkpoint
    revised = copy.deepcopy(old)
    if parameter == "lr":
        revised["optimizer"]["lr"] = 0.005
    elif parameter in ("batch_size", "epoches", "eval_freq", "save_freq"):
        revised["train_params"][parameter] += 3
    elif parameter == "num_workers":
        revised["loader"][parameter] = 0
    elif parameter == "seed":
        revised["seed"] = 917
    model, optimizer, scheduler, generators = runtime_objects(revised)
    state = restore_training(path, model, optimizer, scheduler, generators, provenance(revised), config=revised)
    assert state["epoch"] == 1
    assert all(torch.equal(value, saved["model"][key]) for key, value in model.state_dict().items())
    assert optimizer.state
    assert scheduler.last_epoch == saved["scheduler"]["last_epoch"]
    rate = revised["optimizer"]["lr"] if parameter == "lr" else saved["optimizer"]["param_groups"][0]["lr"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(rate)
    if parameter == "seed":
        assert torch.equal(generators["train"].get_state(), torch.Generator().manual_seed(917).get_state())
        assert torch.equal(generators["validate"].get_state(), torch.Generator().manual_seed(918).get_state())
        assert state["restore_policy"]["rng"].startswith("reseeded")
    else:
        assert state["restore_policy"]["rng"] == "restored"
    optimizer.zero_grad()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(rate)


@pytest.mark.parametrize("change", ["parameters", "type"])
def test_scheduler_changes_restart_current_schedule(training_checkpoint, change):
    path, saved, old = training_checkpoint
    revised = copy.deepcopy(old)
    revised["lr_scheduler"] = ({"core_method": "multistep", "step_size": [4, 6], "gamma": 0.5}
        if change == "parameters" else {"core_method": "StepLR", "step_size": 1, "gamma": 0.8})
    model, optimizer, scheduler, generators = runtime_objects(revised)
    initial_progress = scheduler.last_epoch
    state = restore_training(path, model, optimizer, scheduler, generators, provenance(revised), config=revised)
    assert scheduler.last_epoch == initial_progress
    assert scheduler.gamma == revised["lr_scheduler"]["gamma"]
    assert state["restore_policy"]["scheduler"].startswith("reinitialized")
    assert optimizer.state  # Compatible Adam moments still restored.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(revised["optimizer"]["lr"])
    optimizer.step()
    scheduler.step()
    expected = 0.008 if change == "type" else 0.01
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)


def test_optimizer_switch_safely_reinitializes_state(training_checkpoint):
    path, saved, old = training_checkpoint
    revised = copy.deepcopy(old)
    revised["optimizer"] = {"core_method": "SGD", "lr": 0.02, "args": {"momentum": 0.7}}
    model, optimizer, scheduler, generators = runtime_objects(revised)
    initial_progress = scheduler.last_epoch
    state = restore_training(path, model, optimizer, scheduler, generators, provenance(revised), config=revised)
    assert not optimizer.state
    assert optimizer.param_groups[0]["momentum"] == 0.7
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert scheduler.last_epoch == initial_progress
    assert state["restore_policy"]["optimizer"].startswith("reinitialized")
    before = model.weight.detach().clone()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    assert torch.allclose(model.weight, before - 0.04)


def test_optimizer_argument_changes_safely_reinitialize(training_checkpoint):
    path, _, old = training_checkpoint
    revised = copy.deepcopy(old)
    revised["optimizer"]["args"].update(weight_decay=0.2, betas=[0.7, 0.8])
    model, optimizer, scheduler, generators = runtime_objects(revised)
    restore_training(path, model, optimizer, scheduler, generators, provenance(revised), config=revised)
    assert not optimizer.state
    assert optimizer.param_groups[0]["weight_decay"] == 0.2
    assert list(optimizer.param_groups[0]["betas"]) == [0.7, 0.8]
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()


@pytest.mark.parametrize("has_saved_config", [False, True], ids=["e639-layout", "282-layout"])
def test_legacy_v2xset_checkpoint_ignores_obsolete_metadata(training_checkpoint, has_saved_config):
    path, state, old = training_checkpoint
    state["provenance"].update(config_sha256="unavailable-old-config", code_sha256="old-code",
                               lock_sha256="old-experiment-record")
    if not has_saved_config:
        state.pop("config")
    atomic_torch_save(path, state)
    revised = copy.deepcopy(old)
    revised["optimizer"]["lr"] = 0.004
    model, optimizer, scheduler, generators = runtime_objects(revised)
    restored = restore_training(path, model, optimizer, scheduler, generators, provenance(revised), config=revised)
    assert restored["epoch"] == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.004)
    assert bool(optimizer.state) == has_saved_config
    assert all(torch.equal(value, state["model"][key]) for key, value in model.state_dict().items())
    assert "config" in restored if has_saved_config else "config" not in restored


@pytest.mark.parametrize("mutation", ["shape", "dtype", "keys"])
def test_resume_rejects_incompatible_model(training_checkpoint, mutation):
    path, state, old = training_checkpoint
    model, optimizer, scheduler, generators = runtime_objects(old)
    if mutation == "shape":
        model = torch.nn.Linear(4, 1)
    elif mutation == "dtype":
        state["model"]["weight"] = state["model"]["weight"].double()
    else:
        state["model"].pop("bias")
    atomic_torch_save(path, state)
    with pytest.raises(ValueError, match="model (keys differ|tensor incompatible)"):
        restore_training(path, model, optimizer, scheduler, generators, provenance(old), config=old)


@pytest.mark.parametrize("field", ["dataset", "stage"])
def test_resume_rejects_wrong_dataset_or_stage(training_checkpoint, field):
    path, state, old = training_checkpoint
    state["provenance"][field] = "other"
    atomic_torch_save(path, state)
    model, optimizer, scheduler, generators = runtime_objects(old)
    with pytest.raises(ValueError, match="dataset/stage"):
        restore_training(path, model, optimizer, scheduler, generators, provenance(old), config=old)


def test_parent_checkpoint_identity_ignores_paths_but_not_weights():
    saved = {"dataset": "v2xset", "stage": "code_space/m1",
             "parents": {"single/m1": {"path": "/old/path", "sha256": "weights", "epoch": 3,
                                        "config_sha256": "old-settings"}}}
    expected = copy.deepcopy(saved)
    expected["parents"]["single/m1"].pop("config_sha256")
    expected["parents"]["single/m1"]["path"] = "D:/new/path"
    verify_checkpoint_identity(saved, expected)
    expected["parents"]["single/m1"]["sha256"] = "different-weights"
    with pytest.raises(ValueError, match="parent weights"):
        verify_checkpoint_identity(saved, expected)


def test_corrupted_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "truncated.pth"
    path.write_bytes(b"PK truncated checkpoint")
    with pytest.raises(Exception):
        load_checkpoint(path)


@pytest.mark.parametrize("mutation", ["none", "bytes", "epoch", "dataset"])
def test_parent_checkpoint_integrity_and_stage_dependency(tmp_path, mutation):
    prepare(tmp_path, "v2xset")
    single = load_config(tmp_path / "single/m1/config.yaml")
    child = load_config(tmp_path / "code_space/m1/config.yaml")
    path = tmp_path / "single/m1/best.pth"
    identity = provenance(single)
    state = {"format": "codealign_protocol_v2", "kind": "training", "epoch": 2,
             "model": torch.nn.Linear(3, 1).state_dict(), "provenance": copy.deepcopy(identity)}
    if mutation == "dataset":
        state["provenance"]["dataset"] = "opv2v"
    atomic_torch_save(path, state)
    completion = {"checkpoint_sha256": sha256(path), "best_epoch": 3 if mutation == "epoch" else 2,
                  "provenance": identity}
    (path.parent / "complete.json").write_text(json.dumps(completion), encoding="utf-8")
    if mutation == "bytes":
        path.write_bytes(path.read_bytes() + b"unexpected changes")
    if mutation == "none":
        states, records = parent_checkpoints(tmp_path, child)
        assert list(states) == ["single/m1"]
        assert records["single/m1"]["sha256"] == sha256(path)
        assert set(records["single/m1"]) == {"path", "sha256", "epoch"}
    else:
        with pytest.raises(ValueError):
            parent_checkpoints(tmp_path, child)


@pytest.mark.parametrize("schedule", [
    {"core_method": "LinearLR", "start_factor": 0.1, "total_iters": 5},
    {"core_method": "OneCycleLR", "max_lr": 0.3, "total_steps": 10},
    {"core_method": "CyclicLR", "base_lr": 0.02, "max_lr": 0.2, "step_size_up": 2},
], ids=["linear-warmup", "one-cycle", "cyclic"])
@pytest.mark.parametrize("change", ["schedule", "rate"])
def test_schedule_restart_preserves_current_initialized_rates(training_checkpoint, schedule, change):
    path, state, old = training_checkpoint
    if change == "rate":
        old["lr_scheduler"] = copy.deepcopy(schedule)
        old_model, old_optimizer, old_scheduler, old_generators = runtime_objects(old)
        old_model(torch.ones(2, 3)).sum().backward()
        old_optimizer.step()
        old_scheduler.step()
        state.update(config=copy.deepcopy(old), model=old_model.state_dict(),
                     optimizer=old_optimizer.state_dict(), scheduler=old_scheduler.state_dict(),
                     rng=rng_state(old_generators))
        atomic_torch_save(path, state)
    revised = copy.deepcopy(old)
    revised["lr_scheduler"] = copy.deepcopy(schedule)
    if change == "rate":
        revised["optimizer"]["lr"] = 0.04
    model, optimizer, scheduler, generators = runtime_objects(revised)
    reference_model, reference_optimizer, reference_scheduler, _ = runtime_objects(revised)
    initial_lr = reference_optimizer.param_groups[0]["lr"]
    restored = restore_training(path, model, optimizer, scheduler, generators, provenance(revised), config=revised)
    assert scheduler.last_epoch == reference_scheduler.last_epoch
    assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)
    assert restored["restore_policy"]["effective_learning_rates"] == pytest.approx([initial_lr])
    assert restored["restore_policy"]["configured_optimizer_lr"] == revised["optimizer"]["lr"]
    assert restored["restore_policy"]["scheduler"].startswith("reinitialized")
    # The next update must follow the requested PyTorch schedule, without
    # stale warmup factors, bounds or momentum from the previous checkpoint.
    for _ in range(3):
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
        reference_model(torch.ones(2, 3)).sum().backward()
        reference_optimizer.step()
        reference_scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(reference_optimizer.param_groups[0]["lr"])
        assert optimizer.param_groups[0].get("betas") == reference_optimizer.param_groups[0].get("betas")


@pytest.mark.parametrize("method,argument", [("Adam", "amsgrad"), ("RMSprop", "centered")])
def test_optimizer_state_structure_changes_are_safe_on_next_step(training_checkpoint, method, argument):
    path, state, old = training_checkpoint
    old["optimizer"] = {"core_method": method, "lr": 0.01, "args": {argument: False}}
    old_model, old_optimizer, old_scheduler, old_generators = runtime_objects(old)
    old_model(torch.ones(2, 3)).sum().backward()
    old_optimizer.step()
    old_scheduler.step()
    state.update(config=copy.deepcopy(old), model=old_model.state_dict(), optimizer=old_optimizer.state_dict(),
                 scheduler=old_scheduler.state_dict(), rng=rng_state(old_generators))
    atomic_torch_save(path, state)
    revised = copy.deepcopy(old)
    revised["optimizer"]["args"][argument] = True
    model, optimizer, scheduler, generators = runtime_objects(revised)
    restored = restore_training(path, model, optimizer, scheduler, generators, provenance(revised), config=revised)
    assert not optimizer.state
    assert restored["restore_policy"]["optimizer"].startswith("reinitialized")
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    assert optimizer.state
