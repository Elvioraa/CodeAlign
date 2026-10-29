"""Assignment lock contracts on disposable copies; repository inputs are read-only."""
import copy
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
    # Both datasets exist so cross-dataset reuse is tested against real locks.
    temporary = tempfile.TemporaryDirectory(prefix="codealign-assignment-")
    request.addfinalizer(temporary.cleanup)
    tmp_path = Path(temporary.name)
    root = tmp_path / "repository"
    for dataset in ("opv2v", "v2xset"):
        source = protocol.config_root(dataset)
        target = root / source.relative_to(protocol.ROOT)
        shutil.copytree(source, target)
        for record in protocol.read_yaml(source / "protocol_lock.yaml")["assignments"].values():
            path = root / record["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            value = json.loads((protocol.ROOT / record["path"]).read_text(encoding="utf-8"))
            # Change whitespace only; preserve every scenario/member insertion order.
            path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n\n", encoding="utf-8")
    with patch.object(protocol, "ROOT", root):
        yield request.param, root, tmp_path / "experiment"


def records(dataset):
    return protocol.read_yaml(protocol.config_root(dataset) / "protocol_lock.yaml")["assignments"]


def snapshot(root):
    return {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_prepare_records_observed_bytes_without_changing_inputs(sandbox):
    dataset, root, experiment = sandbox
    before = snapshot(root)
    source = protocol.read_yaml(protocol.config_root(dataset) / "protocol_lock.yaml")
    protocol.prepare(experiment, dataset)
    observed = protocol.read_yaml(experiment / "protocol_lock.yaml")
    expected = copy.deepcopy(source)
    for name, record in records(dataset).items():
        actual = protocol.sha256(root / record["path"])
        assert actual != record["sha256"]
        expected["assignments"][name]["sha256"] = actual
    assert observed == expected
    assert snapshot(root) == before
    protocol.assignments(dataset, experiment_lock=observed)
    # All training/resume/assembly/inference stages share this runtime gate.
    for stage in protocol.STAGES:
        config, _, _ = runtime_config(experiment, stage)
        assert config["codealign_protocol"]["dataset"] == dataset


@pytest.mark.parametrize("mutation", ["scenes", "members", "modality"])
def test_order_or_mapping_changes_are_rejected_before_prepare(sandbox, mutation):
    dataset, root, experiment = sandbox
    path = root / records(dataset)["training"]["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "scenes":
        value = dict(reversed(list(value.items())))
    else:
        scene = next(s for s, members in value.items() if len(members) > 1)
        if mutation == "members":
            value[scene] = dict(reversed(list(value[scene].items())))
        else:
            cav = next(iter(value[scene]))
            value[scene][cav] = "m2" if value[scene][cav] == "m1" else "m1"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="assignment order drift"):
        protocol.prepare(experiment, dataset)
    assert not experiment.exists()


def test_runtime_rejects_new_bytes_even_if_semantics_are_unchanged(sandbox):
    dataset, root, experiment = sandbox
    protocol.prepare(experiment, dataset)
    path = root / records(dataset)["evaluation"]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    # Source audit still accepts its ordered content, but the prepared run cannot.
    protocol.assignments(dataset)
    for stage in protocol.STAGES:
        with pytest.raises(ValueError, match="experiment assignment bytes drift"):
            runtime_config(experiment, stage)
    before = snapshot(experiment)
    with pytest.raises(ValueError, match="experiment assignment bytes drift"):
        protocol.prepare(experiment, dataset)
    assert snapshot(experiment) == before


def test_runtime_rejects_wrong_hash_in_experiment_lock(sandbox):
    dataset, _, experiment = sandbox
    protocol.prepare(experiment, dataset)
    path = experiment / "protocol_lock.yaml"
    value = protocol.read_yaml(path)
    value["assignments"]["training"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="experiment assignment bytes drift"):
        runtime_config(experiment, "single/m1")


def test_prepare_is_idempotent_and_never_overwrites_reviewed_config(sandbox):
    dataset, _, experiment = sandbox
    protocol.prepare(experiment, dataset)
    before = snapshot(experiment)
    protocol.prepare(experiment, dataset)
    assert snapshot(experiment) == before
    path = experiment / "single/m1/config.yaml"
    path.write_bytes(path.read_bytes() + b"\n# reviewed\n")
    before = snapshot(experiment)
    with pytest.raises(ValueError, match="config differs"):
        protocol.prepare(experiment, dataset)
    assert snapshot(experiment) == before


@pytest.mark.parametrize("mutation", ["seed", "ordered", "path", "manifest"])
def test_other_protocol_differences_remain_rejected(sandbox, mutation):
    dataset, _, experiment = sandbox
    protocol.prepare(experiment, dataset)
    path = experiment / ("manifest.yaml" if mutation == "manifest" else "protocol_lock.yaml")
    value = protocol.read_yaml(path)
    if mutation == "seed":
        value["seed"] += 1
    elif mutation == "ordered":
        value["assignments"]["training"]["ordered_sha256"] = "0" * 64
    elif mutation == "path":
        value["assignments"]["training"]["path"] = "different.json"
    else:
        value["dataset"] = "different"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    before = snapshot(experiment)
    with pytest.raises(ValueError):
        protocol.prepare(experiment, dataset)
    assert snapshot(experiment) == before


def test_datasets_cannot_reuse_each_others_experiment(sandbox):
    dataset, _, experiment = sandbox
    protocol.prepare(experiment, dataset)
    other = "v2xset" if dataset == "opv2v" else "opv2v"
    before = snapshot(experiment)
    with pytest.raises(ValueError):
        protocol.prepare(experiment, other)
    with pytest.raises(ValueError):
        protocol.assignments(other, experiment_lock=protocol.read_yaml(experiment / "protocol_lock.yaml"))
    assert snapshot(experiment) == before
