"""Assignment semantics and preparation use disposable copies of current YAML."""
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from opencood.tools import codealign_protocol as protocol
from opencood.tools.run_codealign_doma import runtime_config


@pytest.fixture(params=["opv2v", "v2xset"])
def sandbox(request):
    temporary = tempfile.TemporaryDirectory(prefix="codealign-assignment-")
    request.addfinalizer(temporary.cleanup)
    tmp_path = Path(temporary.name)
    root = tmp_path / "repository"
    for dataset in ("opv2v", "v2xset"):
        source = protocol.config_root(dataset)
        target = root / source.relative_to(protocol.ROOT)
        # Copy only active configuration inputs; no other metadata is needed.
        for relative in ("manifest.yaml", *(f"{stage}/config.yaml" for stage in protocol.STAGES)):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        config = protocol.load_config(source / "single/m1/config.yaml")
        for filename in (config["heter"]["assignment_path"], config["codealign_protocol"]["evaluation_assignment"]):
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            value = json.loads((protocol.ROOT / filename).read_text(encoding="utf-8"))
            path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n\n", encoding="utf-8")
    with patch.object(protocol, "ROOT", root):
        yield request.param, root, tmp_path / "experiment"


def assignment_path(dataset, root, kind):
    config = protocol.load_config(protocol.config_root(dataset) / "single/m1/config.yaml")
    return root / (config["heter"]["assignment_path"] if kind == "training"
                   else config["codealign_protocol"]["evaluation_assignment"])


def snapshot(root):
    return {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_prepare_and_all_runtime_stages_need_only_current_yaml(sandbox):
    dataset, root, experiment = sandbox
    before = snapshot(root)
    protocol.prepare(experiment, dataset)
    assert set(snapshot(experiment)) == {Path("manifest.yaml"), *(Path(stage) / "config.yaml" for stage in protocol.STAGES)}
    assert snapshot(root) == before
    for stage in protocol.STAGES:
        config, _ = runtime_config(experiment, stage)
        assert config["codealign_protocol"]["dataset"] == dataset


def test_format_and_historical_scene_order_are_not_preparation_restrictions(sandbox):
    dataset, root, experiment = sandbox
    path = assignment_path(dataset, root, "training")
    value = json.loads(path.read_text(encoding="utf-8"))
    value = dict(reversed(list(value.items())))
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    protocol.prepare(experiment, dataset)
    assert list(protocol.assignments(dataset)["training"]) == list(value)
    path.write_bytes(path.read_bytes() + b"\n")
    for stage in protocol.STAGES:
        runtime_config(experiment, stage)


@pytest.mark.parametrize("mutation", ["modality", "member", "empty", "prefix", "duplicate", "unknown_scene", "members_coverage"])
def test_invalid_assignment_semantics_rejected_before_prepare(sandbox, mutation):
    dataset, root, experiment = sandbox
    path = assignment_path(dataset, root, "evaluation")
    value = json.loads(path.read_text(encoding="utf-8"))
    scene = next(iter(value))
    members = value[scene]
    cav = next(iter(members))
    if mutation == "modality":
        members[cav] = "m9"
    elif mutation == "member":
        members["not-a-cav"] = members.pop(cav)
    elif mutation == "empty":
        value[scene] = {}
    elif mutation == "prefix":
        members[cav] = "m2"
    elif mutation == "unknown_scene":
        value["unknown-scene"] = value.pop(scene)
    elif mutation == "members_coverage":
        members.pop(next(reversed(members)))
    if mutation == "duplicate":
        text = '{"same-scene":{"1":"m1"},"same-scene":{"1":"m1"}}'
    else:
        text = json.dumps(value)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        protocol.prepare(experiment, dataset)
    assert not experiment.exists()


def test_prepare_preserves_reviewed_yaml_with_all_tunable_settings(sandbox):
    dataset, _, experiment = sandbox
    protocol.prepare(experiment, dataset)
    path = experiment / "single/m1/config.yaml"
    config = protocol.load_config(path)
    config["seed"] = 999
    config["train_params"].update(batch_size=7, epoches=125, eval_freq=3, save_freq=6)
    config["loader"]["num_workers"] = 0
    config["optimizer"] = {"core_method": "SGD", "lr": 0.002, "args": {"momentum": 0.9}}
    config["lr_scheduler"] = {"core_method": "StepLR", "step_size": 10, "gamma": 0.5}
    path.write_text(yaml.safe_dump(config, sort_keys=False) + "\n# reviewed\n", encoding="utf-8")
    before = snapshot(experiment)
    protocol.prepare(experiment, dataset)
    actual, _ = runtime_config(experiment, "single/m1")
    assert actual == config
    assert snapshot(experiment) == before


def test_datasets_cannot_reuse_each_others_experiment(sandbox):
    dataset, _, experiment = sandbox
    protocol.prepare(experiment, dataset)
    other = "v2xset" if dataset == "opv2v" else "opv2v"
    before = snapshot(experiment)
    with pytest.raises(ValueError):
        protocol.prepare(experiment, other)
    config = protocol.load_config(experiment / "single/m1/config.yaml")
    with pytest.raises(ValueError, match="dataset mismatch"):
        protocol.assignments(other, config)
    assert snapshot(experiment) == before


def test_current_yaml_can_select_new_assignment_path(sandbox):
    dataset, root, experiment = sandbox
    protocol.prepare(experiment, dataset)
    path = experiment / "single/m1/config.yaml"
    config = protocol.load_config(path)
    original = root / config["heter"]["assignment_path"]
    chosen = root / "custom-training.json"
    shutil.copyfile(original, chosen)
    config["heter"]["assignment_path"] = str(chosen)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original.unlink()
    actual, _ = runtime_config(experiment, "single/m1")
    assert actual["heter"]["assignment_path"] == str(chosen)


def test_checker_reads_actual_experiment_without_reference_snapshots(sandbox):
    from opencood.tools import check_codealign_doma as checker

    dataset, root, experiment = sandbox
    protocol.prepare(experiment, dataset)
    path = experiment / "single/m1/config.yaml"
    config = protocol.load_config(path)
    config["seed"] += 1
    config["optimizer"] = {"core_method": "SGD", "lr": 0.02, "args": {"momentum": 0.9}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert not (root / "docs").exists()
    with patch.object(checker, "ROOT", Path(__file__).resolve().parents[1]):
        result = checker.static_check(experiment, dataset)
    assert result[dataset]["status"] == "PASS_STATIC"
    assert result[dataset]["stage_configs"] == 12


@pytest.mark.parametrize("mutation", ["stage", "dataset", "manifest_config", "manifest_checkpoint"])
def test_checker_rejects_stage_directory_and_manifest_identity_mismatch(sandbox, mutation):
    from opencood.tools import check_codealign_doma as checker

    dataset, _, experiment = sandbox
    protocol.prepare(experiment, dataset)
    if mutation in ("stage", "dataset"):
        source_dataset = dataset if mutation == "stage" else ("opv2v" if dataset == "v2xset" else "v2xset")
        source_stage = "single/m3" if mutation == "stage" else "single/m2"
        shutil.copyfile(protocol.config_root(source_dataset) / source_stage / "config.yaml",
                        experiment / "single/m2/config.yaml")
        expected = "stage directory/config identity mismatch"
    else:
        path = experiment / "manifest.yaml"
        manifest = protocol.read_yaml(path)
        field = mutation.split("_", 1)[1]
        manifest["stages"]["single/m2"][field] = "single/m3/" + ("config.yaml" if field == "config" else "best.pth")
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        expected = "wrong stage paths"
    with pytest.raises(ValueError, match=expected):
        checker.static_check(experiment, dataset)


def test_generator_emits_only_stage_configs_and_manifest():
    from opencood.tools import generate_codealign_doma_configs as generator

    with tempfile.TemporaryDirectory(prefix="codealign-generate-") as temporary:
        temp = Path(temporary)
        repository, reference = temp / "repository", temp / "reference"
        names = {"m1": "stage1/m1_base", "m2": "stage2/m2_alignto_m1_lr0p0012",
                 "m3": "stage2/m3_alignto_m1_lr0p0030", "m4": "stage2/m4_alignto_m1_lr0p0017",
                 "final": "final_infer"}
        snapshot = {}
        for name, stage in names.items():
            text = (protocol.ROOT / "docs/protocol_sources/opv2v" / f"{name}.yaml").read_text(encoding="utf-8")
            snapshot[f"opencood/logs/DOMA_V2_R2_2GPU_LR1P414_SEED303/{stage}/config.yaml"] = text
        history = reference / "opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2_R2_TABLE4_EXT/history_snapshot.json"
        history.parent.mkdir(parents=True)
        history.write_text(json.dumps(snapshot), encoding="utf-8")
        names = {"m1": "stage1/m1", "m2": "stage2/m2", "m3": "stage2/m3", "m4": "stage2/m4",
                 "final": "final_infer/m1m2m3m4"}
        for name, stage in names.items():
            target = reference / "opencood/hypes_yaml/v2xset/MoreModality/DOMA/V2" / f"{stage}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(protocol.ROOT / "docs/protocol_sources/v2xset" / f"{name}.yaml", target)
        native = Path("opencood/hypes_yaml/opv2v/Heter_group/codebook")
        (repository / native).mkdir(parents=True)
        for filename in ("pyramid.yaml", "codebook_pyramid_fixenc_aligner.yaml", "codebook_pyramid_c2c.yaml"):
            shutil.copyfile(protocol.ROOT / native / filename, repository / native / filename)
        with patch.object(protocol, "ROOT", repository), patch.object(generator, "ROOT", repository):
            generator.generate(reference)
            for dataset in ("opv2v", "v2xset"):
                assert set(snapshot_path.relative_to(protocol.config_root(dataset))
                           for snapshot_path in protocol.config_root(dataset).rglob("*") if snapshot_path.is_file()) == {
                               Path("manifest.yaml"), *(Path(stage) / "config.yaml" for stage in protocol.STAGES)}
            expected = dict(zip(protocol.STAGES[:-1], (4, 2, 8, 4, 8, 4, 8, 4, 2, 2, 2)))
            for stage, batch in expected.items():
                generated = protocol.load_config(protocol.config_root("opv2v") / stage / "config.yaml")
                assert generated["train_params"]["batch_size"] == batch
