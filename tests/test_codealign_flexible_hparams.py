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

from opencood.tools.codealign_protocol import config_root, digest, load_config, prepare, protocol_config, read_yaml, sha256
from opencood.tools.codealign_state import (atomic_torch_save, data_model_fingerprint, create_scheduler, legacy_base_hashes,
                                             provenance, restore_training, rng_state)
from opencood.tools.run_codealign_doma import make_loader, summarize, train_stage


def config():
    return load_config(config_root("v2xset") / "single/m1/config.yaml")


def test_hparams_are_excluded_but_data_and_model_are_pinned():
    original = config()
    changed = copy.deepcopy(original)
    changed["optimizer"] = {"core_method": "SGD", "lr": 0.03, "args": {"momentum": 0.9}}
    changed["lr_scheduler"] = {"core_method": "step", "step_size": 3, "gamma": 0.8}
    changed["train_params"].update(batch_size=7, epoches=88)
    assert protocol_config(original) == protocol_config(changed)
    assert provenance(original, "lock") == provenance(changed, "lock")
    changed["root_dir"] = "dataset/OTHER/train"
    assert provenance(original, "lock") != provenance(changed, "lock")
    changed = copy.deepcopy(original)
    changed["model"]["args"]["channels"] = 32
    assert provenance(original, "lock") != provenance(changed, "lock")


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
                      "checkpoint_sha256": "model", "lock_sha256": "lock", "range": [1], "comm_range": 180,
                      "assignment_sha256": str(n), "assignment_ordered_sha256": "same-order",
                      "use_cav": n, "ap30": 0.1, "ap50": 0.1, "ap70": 0.1}
            (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (path / "sample_ledger.json").write_text(json.dumps([{"modalities": ["m1", "m2", "m3", "m4"][:n]}]), encoding="utf-8")
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
             patch("opencood.tools.run_codealign_doma.device_for_run", return_value=torch.device("cpu")):
            train_stage(root, "single/m1", False)
        assert (root / "single/m1/complete.json").is_file()
        assert (root / "single/m1/latest.pth").is_file()
        assert len(list((root / "single/m1/run_history").glob("*/run.json"))) == 1


def test_legacy_checkpoint_requires_and_uses_recoverable_provenance():
    old = config()
    hashes = [full for full, core in legacy_base_hashes() if core == data_model_fingerprint()]
    assert hashes, "current model/data code differs from legacy base"
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=old["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3], gamma=0.1)
    generators = {"train": torch.Generator().manual_seed(303)}
    revised = copy.deepcopy(old)
    revised["optimizer"]["lr"] = 0.004
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_lock = config_root("v2xset") / "protocol_lock.yaml"
        (root / "protocol_lock.yaml").write_bytes(source_lock.read_bytes())
        path = root / "latest.pth"
        old_identity = {"dataset": "v2xset", "stage": "single/m1", "parents": {},
                        "config_sha256": digest(old), "lock_sha256": sha256(root / "protocol_lock.yaml"),
                        "code_sha256": hashes[0]}
        atomic_torch_save(path, {"format": "codealign_protocol_v2", "kind": "training", "epoch": 1,
                                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scheduler": scheduler.state_dict(), "rng": rng_state(generators),
                                 "provenance": old_identity})
        state = restore_training(path, model, optimizer, scheduler, generators,
                                 provenance(revised, "semantic lock"), config=revised, experiment=root)
        assert state["config"] == old
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.004)
        import yaml
        changed_lock = read_yaml(root / "protocol_lock.yaml")
        changed_lock["seed"] += 1
        (root / "protocol_lock.yaml").write_text(yaml.safe_dump(changed_lock), encoding="utf-8")
        with pytest.raises(ValueError, match="legacy lock provenance"):
            restore_training(path, model, optimizer, scheduler, generators,
                             provenance(revised, "semantic lock"), config=revised, experiment=root)


@pytest.mark.parametrize("change_lr", [False, True])
def test_resume_preserves_scheduler_and_applies_requested_rate(change_lr):
    old = config()
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=old["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3], gamma=0.1)
    generators = {"train": torch.Generator().manual_seed(303)}
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    previous_rate = optimizer.param_groups[0]["lr"]
    revised = copy.deepcopy(old)
    revised["train_params"].update(batch_size=4, epoches=90)
    if change_lr:
        revised["optimizer"]["lr"] = 0.005
    identity = provenance(old, "lock")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "latest.pth"
        atomic_torch_save(path, {"format": "codealign_protocol_v2", "kind": "training", "epoch": 1,
                                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scheduler": scheduler.state_dict(), "rng": rng_state(generators),
                                 "provenance": identity, "config": old})
        new_model = torch.nn.Linear(3, 1)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=revised["optimizer"]["lr"])
        new_scheduler = torch.optim.lr_scheduler.MultiStepLR(new_optimizer, milestones=[1, 3], gamma=0.1)
        restore_training(path, new_model, new_optimizer, new_scheduler, generators,
                         provenance(revised, "lock"), config=revised)
        expected = revised["optimizer"]["lr"] if change_lr else previous_rate
        assert new_optimizer.param_groups[0]["lr"] == pytest.approx(expected)
        new_optimizer.step()
        new_scheduler.step()
        assert new_optimizer.param_groups[0]["lr"] == pytest.approx(expected)
        invalid = copy.deepcopy(revised)
        invalid["lr_scheduler"]["gamma"] = 0.9
        changed_scheduler = torch.optim.lr_scheduler.MultiStepLR(new_optimizer, milestones=[1, 3], gamma=0.9)
        restore_training(path, new_model, new_optimizer, changed_scheduler, generators,
                         provenance(invalid, "lock"), config=invalid)
        assert changed_scheduler.gamma == 0.9
        assert new_optimizer.param_groups[0]["lr"] == pytest.approx(invalid["optimizer"]["lr"])
        changed_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)
        changed_schedule = torch.optim.lr_scheduler.StepLR(changed_optimizer, step_size=3, gamma=0.8)
        switched = copy.deepcopy(revised)
        switched["optimizer"] = {"core_method": "SGD", "lr": 0.01, "args": {"momentum": 0.9}}
        switched["lr_scheduler"] = {"core_method": "step", "step_size": 3, "gamma": 0.8}
        restore_training(path, new_model, changed_optimizer, changed_schedule, generators,
                         provenance(switched, "lock"), config=switched)
        assert changed_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
        assert changed_schedule.last_epoch == 1
        assert not changed_optimizer.state
        incompatible = torch.nn.Linear(4, 1)
        with pytest.raises(ValueError, match="tensor incompatible"):
            restore_training(path, incompatible, new_optimizer, new_scheduler, generators,
                             identity, config=revised)
