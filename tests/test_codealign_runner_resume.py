"""CPU-only integration of the real runner with tiny sensor-free stand-ins."""
import copy
import json
import types
from pathlib import Path

import pytest
import torch
import yaml

from opencood.tools import run_codealign_doma as runner
from opencood.tools.codealign_protocol import assignments, load_config, prepare, sha256
from opencood.tools.codealign_state import atomic_torch_save, load_checkpoint


@pytest.fixture
def interrupted_stage(tmp_path, monkeypatch):
    import opencood.tools

    scenes = list(assignments("v2xset")["training"])[:2]
    control = {"interrupt": True, "changed_samples": False}

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, train): self.train = train
        def __len__(self): return 8
        def __getitem__(self, index): return float(index) / 8
        def reinitialize(self): pass
        def collate_batch_train(self, batch):
            x = torch.tensor(batch).reshape(-1, 1)
            return {"ego": {"x": x, "label_dict": {"y": x}, "record_len": torch.ones(len(batch))}}

    class Model(torch.nn.Linear):
        def __init__(self): super().__init__(1, 1)
        def model_train_init(self): pass
        def forward(self, data):
            if control["interrupt"] and self.training and data["epoch"] == 1:
                raise RuntimeError("deliberate CPU interruption")
            return {"prediction": super().forward(data["x"])}

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
    fake_utils.to_device = lambda batch, device: batch
    dataset_module = types.ModuleType("opencood.data_utils.datasets.codealign_protocol_dataset")
    dataset_module.build_protocol_dataset = lambda hypes, train: Dataset(train)
    dataset_module.sample_keys = lambda data: [
        [scenes[0 if data.train else 1], str(i + (100 if control["changed_samples"] else 0))]
        for i in range(8)
    ]
    datasets = types.ModuleType("opencood.data_utils.datasets")
    datasets.__path__ = []
    writer_module = types.ModuleType("tensorboardX")
    writer_module.SummaryWriter = Writer
    for name, module in {"opencood.tools.train_utils": fake_utils, "tensorboardX": writer_module,
                         "opencood.data_utils.datasets": datasets,
                         "opencood.data_utils.datasets.codealign_protocol_dataset": dataset_module}.items():
        monkeypatch.setitem(__import__("sys").modules, name, module)
    monkeypatch.setattr(opencood.tools, "train_utils", fake_utils, raising=False)
    monkeypatch.setattr(runner, "device_for_run", lambda: torch.device("cpu"))
    prepare(tmp_path, "v2xset")
    directory = tmp_path / "single/m1"
    config_path = directory / "config.yaml"
    config = load_config(config_path)
    config["train_params"].update(batch_size=2, epoches=3, eval_freq=1, save_freq=1)
    config["loader"].update(num_workers=0, pin_memory=False)
    config["lr_scheduler"] = {"core_method": "multistep", "step_size": [2, 4], "gamma": 0.1}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="deliberate CPU interruption"):
        runner.train_stage(tmp_path, "single/m1", False)
    assert load_checkpoint(directory / "latest.pth")["epoch"] == 1
    assert not (directory / "complete.json").exists()
    history = next((directory / "run_history").iterdir())
    assert "deliberate CPU interruption" in (history / "run.log").read_text()
    assert json.loads((history / "outcome.json").read_text())["status"] == "failed"
    control["interrupt"] = False
    return tmp_path, directory, config, control


@pytest.mark.parametrize("change", ["lr", "batch", "epochs", "scheduler", "optimizer", "seed", "frequencies"])
def test_current_yaml_controls_resume_and_preserves_history(interrupted_stage, change):
    root, directory, previous, control = interrupted_stage
    revised = copy.deepcopy(previous)
    if change == "lr": revised["optimizer"]["lr"] = 0.004
    if change == "batch": revised["train_params"]["batch_size"] = 4
    if change == "epochs": revised["train_params"]["epoches"] = 4
    if change == "scheduler": revised["lr_scheduler"] = {"core_method": "step", "step_size": 8, "gamma": 0.8}
    if change == "optimizer": revised["optimizer"] = {"core_method": "SGD", "lr": 0.02, "args": {"momentum": 0.9}}
    if change == "seed": revised["seed"] = 707
    if change == "frequencies": revised["train_params"].update(eval_freq=2, save_freq=2)
    audit_before = (directory / "input_audit.json").read_bytes()
    checkpoint_before = sha256(directory / "latest.pth")
    epoch_before = sha256(directory / "epoch_0001.pth")
    (directory / "config.yaml").write_text(yaml.safe_dump(revised), encoding="utf-8")
    runner.train_stage(root, "single/m1", True)
    state = load_checkpoint(directory / "latest.pth")
    assert state["epoch"] == revised["train_params"]["epoches"]
    assert state["config"] == revised
    assert (directory / "input_audit.json").read_bytes() == audit_before
    assert sha256(directory / "epoch_0001.pth") == epoch_before
    records = list((directory / "run_history").glob("*/run.json"))
    assert len(records) == 2
    resumed = next(path for path in records if json.loads(path.read_text())["resume"])
    record = json.loads(resumed.read_text())
    assert record["config"] == revised and record["previous_config"] == previous
    assert record["effective_learning_rates"] == pytest.approx([revised["optimizer"]["lr"]])
    assert sha256(resumed.parent / "latest.pth") == checkpoint_before
    if change == "scheduler": assert record["scheduler_last_epoch"] == 0
    if change == "optimizer": assert record["optimizer_type"] == "SGD"
    with pytest.raises(ValueError, match="stage already complete"):
        runner.train_stage(root, "single/m1", True)


@pytest.mark.parametrize("change", ["samples", "members"])
def test_resume_rejects_dataset_identity_change(interrupted_stage, monkeypatch, change):
    root, directory, config, control = interrupted_stage
    if change == "samples":
        control["changed_samples"] = True
    else:
        changed = assignments("v2xset", config=config)
        first = next(iter(changed["training"].values()))
        first[next(iter(first))] = "m4"
        monkeypatch.setattr(runner, "assignments", lambda *args, **kwargs: changed)
    checkpoint_before = sha256(directory / "latest.pth")
    with pytest.raises(ValueError, match="data population changed|sample members changed"):
        runner.train_stage(root, "single/m1", True)
    assert sha256(directory / "latest.pth") == checkpoint_before
    assert not (directory / "complete.json").exists()


def test_existing_weights_require_explicit_resume(interrupted_stage):
    root, directory, _, _ = interrupted_stage
    checkpoint_before = sha256(directory / "latest.pth")
    with pytest.raises(ValueError, match="existing checkpoint"):
        runner.train_stage(root, "single/m1", False)
    assert sha256(directory / "latest.pth") == checkpoint_before


def test_legacy_v2xset_checkpoint_and_audit_resume_without_old_settings(interrupted_stage):
    root, directory, config, _ = interrupted_stage
    state = load_checkpoint(directory / "latest.pth")
    state.pop("config")
    state.pop("sample_members")
    # Old metadata is deliberately wrong; only dataset/stage/parents and
    # actual sample identity are relevant to loading these legacy weights.
    state["provenance"].update(config_sha256="old-config", code_sha256="old-source", lock_sha256="old-lock")
    atomic_torch_save(directory / "latest.pth", state)
    audit_path = directory / "input_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["provenance"] = state["provenance"]
    audit.pop("sample_members")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_before = audit_path.read_bytes()
    config["optimizer"]["lr"] = 0.003
    config["train_params"].update(batch_size=4, epoches=4)
    (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    runner.train_stage(root, "single/m1", True)
    assert load_checkpoint(directory / "latest.pth")["epoch"] == 4
    assert audit_path.read_bytes() == audit_before
    records = [json.loads(path.read_text()) for path in (directory / "run_history").glob("*/run.json")]
    resumed = next(record for record in records if record["resume"])
    assert resumed["previous_config"] is None
    assert resumed["effective_learning_rates"] == pytest.approx([0.003])
    assert "reinitialized" in resumed["restore_policy"]["optimizer"]


@pytest.mark.parametrize("mutation", [None, "bytes", "dataset", "missing_record"])
def test_assembled_checkpoint_integrity(tmp_path, mutation):
    identity = {"dataset": "v2xset", "stage": "final_infer", "parents": {}}
    path = tmp_path / "best.pth"
    atomic_torch_save(path, {"format": "codealign_protocol_v2", "kind": "assembly", "epoch": 0,
                             "model": torch.nn.Linear(1, 1).state_dict(), "provenance": identity})
    record = {"checkpoint_sha256": sha256(path), "provenance": copy.deepcopy(identity)}
    if mutation == "bytes": path.write_bytes(path.read_bytes() + b"corrupt-tail")
    if mutation == "dataset": record["provenance"]["dataset"] = "opv2v"
    if mutation != "missing_record":
        (tmp_path / "assembly.json").write_text(json.dumps(record), encoding="utf-8")
    if mutation is None:
        assert runner.load_assembly(tmp_path, identity)["kind"] == "assembly"
    else:
        with pytest.raises(ValueError, match="assembled checkpoint changed|dataset/stage drift|missing assembly"):
            runner.load_assembly(tmp_path, identity)
