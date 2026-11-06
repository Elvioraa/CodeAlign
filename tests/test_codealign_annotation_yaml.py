"""Frame annotations use the upstream reader; experiment YAML stays safe."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from opencood.hypes_yaml import yaml_utils
from opencood.tools import check_codealign_doma as checker
from opencood.tools.codealign_protocol import ConfigLoader, ROOT, config_root, load_config, read_yaml


def annotation():
    return {"lidar_pose": [0.0] * 6, "vehicles": {},
            "plan_trajectory": [[np.float64(1.25), np.float32(-2.5)]],
            **{f"camera{i}": {"cords": [0.0] * 6, "intrinsic": np.eye(3).tolist()} for i in range(4)}}


def numpy_yaml():
    # Exercise the exact legacy module name found in original OPV2V files,
    # including when the test environment uses NumPy 2's newer module name.
    return yaml.dump(annotation()).replace("numpy._core.multiarray", "numpy.core.multiarray")


def test_plan_trajectory_numpy_scalars_use_official_reader(tmp_path):
    path = tmp_path / "000068.yaml"
    path.write_text(numpy_yaml(), encoding="utf-8")
    assert "python/object/apply:numpy.core.multiarray.scalar" in path.read_text()
    with patch.object(yaml_utils, "load_yaml", wraps=yaml_utils.load_yaml) as official:
        value = checker.read_dataset_yaml(path)
    official.assert_called_once_with(str(path))
    x, y = value["plan_trajectory"][0]
    assert isinstance(x, np.float64) and x == 1.25
    assert isinstance(y, np.float32) and y == -2.5


def test_config_reader_remains_safe_after_annotation_loading(tmp_path):
    path = tmp_path / "annotation.yaml"
    path.write_text(numpy_yaml(), encoding="utf-8")
    checker.read_dataset_yaml(path)
    assert issubclass(ConfigLoader, yaml.SafeLoader)
    with pytest.raises(yaml.constructor.ConstructorError):
        read_yaml(path)
    with patch("opencood.tools.codealign_protocol.yaml.load", wraps=yaml.load) as safe_read:
        config = load_config(config_root("opv2v") / "single/m3/config.yaml")
    assert safe_read.call_args.kwargs["Loader"] is ConfigLoader
    assert config["codealign_protocol"]["stage"] == "single/m3"


def test_annotation_import_and_parse_do_not_import_torch_or_cuda(tmp_path):
    path = tmp_path / "annotation.yaml"
    path.write_text(numpy_yaml(), encoding="utf-8")
    program = """
import sys
from opencood.tools.check_codealign_doma import read_dataset_yaml
value = read_dataset_yaml(sys.argv[1])
assert float(value['plan_trajectory'][0][0]) == 1.25
assert not any(name == 'torch' or name.startswith(('torch.', 'cupy', 'pycuda')) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", program, str(path)], cwd=ROOT, check=True,
                   capture_output=True, text=True)


@pytest.mark.parametrize("dataset", ["opv2v", "v2xset"])
@pytest.mark.parametrize("json_first", [False, True])
def test_audit_annotations_for_both_datasets(tmp_path, dataset, json_first):
    root = tmp_path / ("OPV2V" if dataset == "opv2v" else "V2XSET")
    assigned = {"training": {}, "evaluation": {}}
    for split in ("train", "validate", "test"):
        scene = "scene_" + split
        assigned["training"][scene] = {"10710": "m1"}
        if split == "test": assigned["evaluation"][scene] = {"10710": "m1"}
        directory = root / split / scene / "10710"
        directory.mkdir(parents=True)
        prefix = directory / "000068"
        prefix.with_suffix(".yaml").write_text("not: [valid yaml" if json_first else numpy_yaml(), encoding="utf-8")
        if json_first:
            value = annotation()
            value["plan_trajectory"] = [[1.25, -2.5]]
            prefix.with_suffix(".json").write_text(json.dumps(value), encoding="utf-8")
        prefix.with_suffix(".pcd").write_text("FIELDS x y z intensity\nDATA ascii\n", encoding="ascii")
        for camera in range(4):
            Path(str(prefix) + f"_camera{camera}.png").touch()
        if dataset == "opv2v":
            extra = tmp_path / "OPV2V_Hetero" / split / scene / "10710" / "000068"
            extra.parent.mkdir(parents=True)
            Path(str(extra) + "_32.pcd").touch()
            for camera in range(4):
                Path(str(extra) + f"_depth{camera}.png").touch()
            Path(str(prefix) + "_bev_visibility.png").touch()
    with patch.object(checker, "assignments", return_value=assigned), \
         patch.object(yaml_utils, "load_yaml", wraps=yaml_utils.load_yaml) as official:
        result = checker.audit_data(dataset, data_root=root, frames_per_scene=2)
    assert result["status"] == "PASS_SAMPLED_DATA_SCAN"
    assert result["checked_CAV_frames"] == 3
    assert official.call_count == (0 if json_first else 3)


def test_original_opv2v_000068_when_available():
    path = ROOT / "dataset/OPV2V/train/2021_08_23_16_42_39/10710/000068.yaml"
    if not path.is_file():
        pytest.skip("Original OPV2V 000068.yaml is not available locally")
    value = checker.read_dataset_yaml(path)
    assert "lidar_pose" in value and "vehicles" in value

    def scalars(item):
        if isinstance(item, dict):
            return [scalar for child in item.values() for scalar in scalars(child)]
        if isinstance(item, (list, tuple)):
            return [scalar for child in item for scalar in scalars(child)]
        return [item] if isinstance(item, np.generic) else []

    assert scalars(value["plan_trajectory"]), "Expected NumPy scalars in the reported annotation"
