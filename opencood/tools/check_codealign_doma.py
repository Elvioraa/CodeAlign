"""Static gates, optional real-data audit and CUDA smoke (never trains)."""
import argparse
import ast
import copy
import json
from pathlib import Path

from opencood.tools.codealign_protocol import (
    ROOT, MODALITIES, STAGES, assignments, config_root, digest, evaluation_config,
    load_config, read_yaml, require, scene_owner, sha256, stage_parents, write_json,
)


def static_check():
    from opencood.hypes_yaml.yaml_utils import load_general_params
    results = {}
    audit = json.loads((ROOT / "docs/protocol_sources/code_implementation_audit.json").read_text(encoding="utf-8"))
    for record in audit["records"]:
        node = ast.parse((ROOT / record["file"]).read_text(encoding="utf-8"))
        for part in record["symbol"].split("."):
            node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
        import hashlib
        require(hashlib.sha256(ast.dump(node).encode()).hexdigest() == record["codealign_ast_sha256"],
                f"audited native function changed: {record['symbol']}")
    for dataset in ("opv2v", "v2xset"):
        directory = config_root(dataset)
        lock = read_yaml(directory / "protocol_lock.yaml")
        assignment = assignments(dataset)
        manifest = read_yaml(directory / "manifest.yaml")
        require(list(manifest["stages"]) == list(STAGES), "incomplete stage graph")
        for source in lock["reference_configs"].values():
            require(sha256(ROOT / source["copy"]) == source["sha256"], "reference snapshot changed")
        shard_scenes = {m: {s for s in assignment["training"] if scene_owner(s, assignment["training"]) == m}
                        for m in MODALITIES}
        require(set.union(*shard_scenes.values()) == set(assignment["training"]), "unassigned scene")
        for i, first in enumerate(MODALITIES):
            for second in MODALITIES[i+1:]:
                require(not shard_scenes[first] & shard_scenes[second], "modality isolation leakage")
        for stage in STAGES:
            config = load_config(directory / stage / "config.yaml")
            entry = manifest["stages"][stage]
            require(entry["parents"] == stage_parents(stage), f"wrong parent graph: {stage}")
            for parent in entry["parents"]:
                require(STAGES.index(parent) < STAGES.index(stage), "cycle in stage graph")
                require((directory / manifest["stages"][parent]["config"]).is_file(), "missing parent config")
            require((ROOT / config["codealign_protocol"]["budget_source"]).is_file(), "unknown budget source")
            parsed = load_general_params(copy.deepcopy(config))
            anchor = parsed["postprocess"]["anchor_args"]
            args = parsed["model"]["args"]
            require([anchor["H"]//2, anchor["W"]//2] == args["translator"]["args"]["out_feature_shape"],
                    "anchor / BEV shape mismatch")
            require(isinstance(config["optimizer"]["args"]["eps"], float), "optimizer epsilon parsed as string")
            ref = read_yaml(ROOT / config["codealign_protocol"]["reference_sensor_source"])
            for m, data in config["heter"]["modality_setting"].items():
                expected = copy.deepcopy(ref["heter"]["modality_setting"][m])
                if stage == "final_infer" and m in ("m1", "m3"):
                    expected["preprocess"]["cav_lidar_range"] = config["cav_lidar_range"]
                require(data == expected, f"sensor preprocessing drift: {dataset}/{stage}/{m}")
                expected_model = copy.deepcopy(ref["model"]["args"][m])
                expected_model.pop("aligner_args", None)
                if stage == "final_infer" and m in ("m1", "m3"):
                    expected_model["encoder_args"]["lidar_range"] = config["cav_lidar_range"]
                require(args["modalities"][m] == expected_model, f"encoder architecture drift: {stage}/{m}")
        final = load_config(directory / "final_infer/config.yaml")
        for cav in (1, 2, 3, 4):
            selected = evaluation_config(final, cav)
            require(selected["use_cav"] == cav and selected["validate_dir"] == selected["test_dir"], "bad evaluation split")
            require(selected["heter"]["assignment_path"] == lock["assignments"]["evaluation"]["path"], "wrong ordered assignment")
        # Confirm inherited AP/NMS implementations against copies' source repo
        # is recorded separately; NEVER replace the official metric here.
        results[dataset] = {"status": "PASS_STATIC", "stage_configs": len(STAGES),
                            "training_assignment_scenes": len(assignment["training"]),
                            "evaluation_assignment_scenes": len(assignment["evaluation"]),
                            "isolation_scene_counts": {m: len(s) for m, s in shard_scenes.items()},
                            "real_data": "SERVER_REQUIRED", "full_model": "SERVER_REQUIRED",
                            "final_runtime_protocol": "SERVER_VALIDATION_REQUIRED"}
    files = [ROOT / "opencood/models/codealign_doma_protocol.py",
             ROOT / "opencood/data_utils/datasets/codealign_protocol_dataset.py",
             ROOT / "opencood/utils/codealign_data.py"]
    files += list((ROOT / "opencood/tools").glob("*codealign*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    results["python_ast_files"] = len(files)
    return results


def pcd_fields(path):
    with Path(path).open("rb") as stream:
        for _ in range(100):
            line = stream.readline().decode("ascii", errors="replace").strip()
            if line.upper().startswith("FIELDS "):
                return line.split()[1:]
            if line.upper().startswith("DATA "):
                break
    raise ValueError(f"missing PCD fields: {path}")


def audit_data(dataset, data_root=None, frames_per_scene=0):
    """Scan counts/coverage; 0 checks every timestamp, positive N is sampled."""
    require(frames_per_scene >= 0, "frames_per_scene must be nonnegative")
    lock = read_yaml(config_root(dataset) / "protocol_lock.yaml")
    assigned = assignments(dataset)
    root = Path(data_root) if data_root else ROOT / Path(lock["splits"]["root_dir"]).parent
    split_dirs = {split: (root / split).resolve() for split in ("train", "validate", "test")}
    for path in split_dirs.values():
        require(path.is_dir(), f"SERVER_REQUIRED: data directory missing: {path}")
    require(len(set(split_dirs.values())) == 3, "split symlinks resolve to same directory")
    scenes_by_split, output, checked = {}, {}, 0
    for split, directory in split_dirs.items():
        scenes = sorted(p for p in directory.iterdir() if p.is_dir())
        scenes_by_split[split] = {p.name for p in scenes}
        frames = 0
        shard_counts = dict.fromkeys(MODALITIES, 0)
        for scene in scenes:
            require(scene.name in assigned["training"], f"assignment lacks real scene: {split}/{scene.name}")
            if scene.name in lock["camera_excluded_scenes"]:
                continue
            members = assigned["evaluation" if split == "test" else "training"].get(scene.name)
            require(members is not None, f"test-only ordered assignment missing scene: {scene.name}")
            actual = {p.name for p in scene.iterdir() if p.is_dir()}
            require(set(members) == actual, f"invalid/placeholder/missing CAV assignment: {scene.name}: {set(members)^actual}")
            timestamps = sorted(p.stem for p in (scene / next(iter(members))).glob("*.yaml") if "additional" not in p.name)
            timestamps = [t for t in timestamps if not (scene.name == "2021_08_20_21_10_24" and t == "000265")]
            require(timestamps, f"no frames: {scene}")
            frames += len(timestamps)
            shard_counts[scene_owner(scene.name, assigned["training"])] += len(timestamps)
            for timestamp in (timestamps[:frames_per_scene] if frames_per_scene else timestamps):
                for cav in members:
                    prefix = scene / cav / timestamp
                    require(prefix.with_suffix(".yaml").is_file(), f"missing timestamp metadata: {prefix}")
                    metadata_json = prefix.with_suffix(".json")
                    metadata = json.loads(metadata_json.read_text()) if metadata_json.is_file() else read_yaml(prefix.with_suffix(".yaml"))
                    require("lidar_pose" in metadata and "vehicles" in metadata, f"missing pose/labels: {prefix}")
                    original = prefix.with_suffix(".pcd")
                    require(original.is_file(), f"missing original PCD: {original}")
                    fields = pcd_fields(original)
                    require({"x", "y", "z"} <= set(fields) and bool({"intensity", "rgb", "rgba"} & set(fields)),
                            f"cannot construct four point features: {original}")
                    image_pack = Path(str(prefix) + "_imgs.hdf5")
                    packed = dataset == "opv2v" and image_pack.is_file()
                    if packed:
                        import h5py
                        with h5py.File(image_pack, "r") as stream:
                            require(all(f"{sensor}{camera}" in stream for sensor in ("camera", "depth") for camera in range(4)),
                                    f"HDF5 lacks required RGB/depth: {image_pack}")
                    for camera in range(4):
                        if not packed:
                            require(Path(str(prefix) + f"_camera{camera}.png").is_file(), f"missing RGB camera: {prefix}/{camera}")
                        calib = metadata.get(f"camera{camera}", {})
                        require("cords" in calib and "intrinsic" in calib, f"missing camera calibration: {prefix}")
                    if dataset == "opv2v":
                        parts = list(prefix.parts)
                        require("OPV2V" in parts, "OPV2V extension directory identity is ambiguous")
                        parts[parts.index("OPV2V")] = "OPV2V_Hetero"
                        extra = Path(*parts)
                        require(Path(str(extra) + "_32.pcd").is_file(), f"missing true 32-beam extension: {extra}")
                        if not packed:
                            for camera in range(4):
                                require(Path(str(extra) + f"_depth{camera}.png").is_file(), f"missing OPV2V-H depth: {extra}")
                        visibility = Path(str(prefix) + "_bev_visibility.png")
                        alternate = root / "additional" / split / scene.name / cav / (timestamp + "_bev_visibility.png")
                        require(visibility.is_file() or alternate.is_file(), f"missing visibility label: {prefix}")
                    checked += 1
        output[split] = {"scenes": len(scenes), "included_scenes": len(scenes_by_split[split] - set(lock["camera_excluded_scenes"])),
                         "frames": frames, "source_shard_frames": shard_counts}
    for first, second in (("train", "validate"), ("train", "test"), ("validate", "test")):
        require(not scenes_by_split[first] & scenes_by_split[second], f"scene leakage: {first}/{second}")
    require(set(assigned["evaluation"]) == scenes_by_split["test"], "ordered assignment test scene coverage differs")
    require(set(assigned["training"]) == set.union(*scenes_by_split.values()),
            "training assignment contains unknown/placeholder scenes or lacks real scenes")
    return {"status": "PASS_DATA_SCAN" if frames_per_scene == 0 else "PASS_SAMPLED_DATA_SCAN",
            "splits": output, "checked_CAV_frames": checked, "frames_per_scene": frames_per_scene,
            "physical_beam_provenance": "OPV2V _32 file existence checked; beam production provenance still needs dataset records"}


def cuda_smoke(dataset, stage, experiment=None):
    import torch
    from opencood.tools.run_codealign_doma import device_for_run, make_loader
    device = device_for_run()
    from opencood.data_utils.datasets.codealign_protocol_dataset import build_protocol_dataset, sample_keys
    from opencood.tools import train_utils
    from opencood.tools.codealign_state import seed_everything
    from opencood.hypes_yaml.yaml_utils import load_general_params
    path = (Path(experiment) if experiment else config_root(dataset)) / stage / "config.yaml"
    config = load_config(path)
    seed_everything(config["seed"], config["reproducibility"])
    hypes = load_general_params(copy.deepcopy(config))
    evaluation = stage == "final_infer"
    if evaluation:
        hypes = evaluation_config(hypes, 4)
    data = build_protocol_dataset(hypes, train=False, evaluation=evaluation)
    loader = make_loader(data, config, torch.Generator().manual_seed(config["seed"]), evaluation=evaluation)
    batch = next(iter(loader))
    require(batch is not None, "invalid real smoke sample")
    model = train_utils.create_model(hypes, train_flag=False).to(device).eval()
    # Shape smoke is explicitly RANDOM INITIALIZATION, never a trained result.
    with torch.no_grad():
        output = model(train_utils.to_device(batch, device)["ego"])
    require(output["cls_preds"].isfinite().all().item(), "nonfinite smoke forward")
    return {"status": "PASS_CUDA_SHAPE_SMOKE_RANDOM_WEIGHTS", "dataset": dataset, "stage": stage,
            "population": len(data), "output_shape": list(output["cls_preds"].shape),
            "path_counts": dict(model.path_counts), "training_and_AP": "NOT_TESTED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", choices=("opv2v", "v2xset"))
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--frames-per-scene", type=int, default=0)
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default="final_infer")
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    result = static_check()
    if args.check_data:
        require(args.dataset, "data audit needs --dataset")
        result["data_audit"] = audit_data(args.dataset, args.data_root, args.frames_per_scene)
    if args.cuda_smoke:
        require(args.dataset, "smoke needs --dataset")
        result["cuda_smoke"] = cuda_smoke(args.dataset, args.stage, args.experiment_root)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
