"""Dependency-light protocol, provenance and stage graph checks (no CUDA imports)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODALITIES = ("m1", "m2", "m3", "m4")
STAGES = tuple(f"single/{m}" for m in MODALITIES) + tuple(
    f"code_space/{m}" for m in MODALITIES
) + tuple(f"fcf/{m}_to_m1" for m in MODALITIES[1:]) + ("final_infer",)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     sort_keys=True).encode()).hexdigest()


def ordered_hash(value):
    # Assignment member order is meaningful: never sort its keys.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def config_root(dataset):
    require(dataset in ("opv2v", "v2xset"), "unknown dataset")
    return ROOT / "opencood/hypes_yaml" / dataset / "doma_protocol_codealign_v2"


def scene_owner(scene, assignment):
    # The original Adaptor_group divides insertion-ordered scenes round-robin.
    return MODALITIES[list(assignment).index(scene) % len(MODALITIES)]


def assignments(dataset):
    lock = read_yaml(config_root(dataset) / "protocol_lock.yaml")
    result = {}
    for name, record in lock["assignments"].items():
        path = ROOT / record["path"]
        value = json.loads(path.read_text(encoding="utf-8"))
        require(sha256(path) == record["sha256"], f"assignment bytes drift: {path}")
        require(ordered_hash(value) == record["ordered_sha256"], "assignment order drift")
        for scene, members in value.items():
            require(members and all(m in MODALITIES for m in members.values()),
                    f"invalid modality in {scene}")
            require(all(str(int(cav)) == cav for cav in members), f"invalid CAV id in {scene}")
            if name == "evaluation":
                require(next(iter(members.values())) == "m1", f"ego is not m1: {scene}")
                require(int(next(iter(members))) >= 0, f"infrastructure ego: {scene}")
                require(list(members.values())[:4] == list(MODALITIES[:min(4, len(members))]),
                        f"invalid ordered prefix: {scene}")
        result[name] = value
    require(set(result["evaluation"]) <= set(result["training"]), "unknown evaluation scene")
    return result


def load_config(path):
    config = read_yaml(path)
    validate_config(config)
    return config


def validate_config(config):
    protocol = config["codealign_protocol"]
    dataset, stage = protocol["dataset"], protocol["stage"]
    require(dataset in ("opv2v", "v2xset") and stage in STAGES, "invalid stage identity")
    require(config["model"]["core_method"] == "codealign_doma_protocol", "wrong model entry")
    require(config["fusion"]["dataset"] == dataset, "dataset identity mismatch")
    require(config["seed"] == protocol["seed"], "seed mismatch")
    require(len({config[k] for k in ("root_dir", "validate_dir", "test_dir")}) == 3,
            "train/validation/test paths overlap")
    require(config["train_params"]["max_cav"] == 5, "GT max_cav drift")
    args = config["model"]["args"]
    require(args["stage"] == stage and args["dataset"] == dataset, "model/stage mismatch")
    require(args["channels"] == 64, "CodeAlign codebook requires 64 channels")
    require(args["codebook"] == {"seg_num": 1, "dict_size": 16, "r": 1},
            "FCF discrete decoder requires a single 16-entry codebook")
    active = args["modalities"]
    require(list(active) == (list(MODALITIES) if stage == "final_infer" else [protocol["source"]]),
            "active modalities do not match the isolation graph")
    expected = {"m1": ("lidar", "point_pillar"), "m2": ("camera", "lift_splat_shoot"),
                "m3": ("lidar", "second"), "m4": ("camera", "lift_splat_shoot")}
    for m, setting in active.items():
        require((setting["sensor_type"], setting["core_method"]) == expected[m], f"wrong encoder: {m}")
        require(setting["backbone_args"]["num_filters"][-1] == 64, "feature channels mismatch")
        if m in ("m2", "m4"):
            encoder = setting["encoder_args"]
            require(encoder["camera_encoder"] == ("EfficientNet" if m == "m2" else "Resnet101"),
                    f"camera backbone mismatch: {m}")
            require(encoder["depth_supervision"] == (dataset == "opv2v"), "depth contract drift")
        if m == "m3":
            encoder = setting["encoder_args"]
            require(encoder["voxel_size"] == [0.1, 0.1, 0.1], "SECOND voxelization drift")
            require(encoder["mean_vfe"]["num_point_features"] == 4 and
                    encoder["spconv"]["num_features_in"] == 4, "SECOND needs XYZI")
    channels = config["heter"].get("lidar_channels_dict", {})
    require(channels == ({"m3": 32} if dataset == "opv2v" else {}), "dataset-specific PCD policy drift")
    require(isinstance(config["comm_range"], (int, float)) and config["comm_range"] > 0,
            "communication range must be an explicit positive YAML value")
    require(args["translator"]["args"]["in_channels"] == 64 and
            args["translator"]["args"]["out_channels"] == 16, "FCF channel mismatch")
    require(args["translator"]["args"]["in_feature_shape"] ==
            args["translator"]["args"]["out_feature_shape"], "FCF grid scale must be 1")
    require(config["yaml_parser"] == "load_general_params", "unexpected parser")
    return True


def stage_parents(stage):
    if stage.startswith("single/"):
        return []
    if stage.startswith("code_space/"):
        return ["single/" + stage.split("/")[1]]
    if stage.startswith("fcf/"):
        return ["code_space/" + stage.split("/")[1].split("_")[0], "code_space/m1"]
    return [*(f"code_space/{m}" for m in MODALITIES),
            *(f"fcf/{m}_to_m1" for m in MODALITIES[1:])]


def prepare(experiment, dataset):
    experiment = Path(experiment).resolve()
    source = config_root(dataset)
    manifest = read_yaml(source / "manifest.yaml")
    for filename in ("protocol_lock.yaml", "manifest.yaml"):
        dst = experiment / filename
        require(not dst.exists() or dst.read_bytes() == (source / filename).read_bytes(),
                f"existing experiment identity differs: {dst}")
    # Check everything first; never overwrite a reviewed config.
    for stage in STAGES:
        src = source / stage / "config.yaml"
        load_config(src)
        dst = experiment / stage / "config.yaml"
        require(not dst.exists() or dst.read_bytes() == src.read_bytes(), f"config differs: {dst}")
    for stage in STAGES:
        dst = experiment / stage / "config.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes((source / stage / "config.yaml").read_bytes())
    for filename in ("protocol_lock.yaml", "manifest.yaml"):
        dst = experiment / filename
        require(not dst.exists() or dst.read_bytes() == (source / filename).read_bytes(),
                f"existing experiment identity differs: {dst}")
        if not dst.exists():
            dst.write_bytes((source / filename).read_bytes())
    return manifest


def evaluation_config(config, use_cav):
    require(type(use_cav) is int and 1 <= use_cav <= 4, "use_cav must be 1..4")
    config = copy.deepcopy(config)
    require(config["codealign_protocol"]["stage"] == "final_infer", "inference needs final config")
    config["validate_dir"] = config["test_dir"]
    config["use_cav"] = use_cav
    config["fusion"]["core_method"] = "intermediateheterinfer"
    config["heter"]["assignment_path"] = config["codealign_protocol"]["evaluation_assignment"]
    return config


def validate_prefix(modalities, use_cav):
    require(1 <= use_cav <= 4 and 0 < len(modalities) <= use_cav, "invalid CAV count")
    require(list(modalities) == list(MODALITIES[:len(modalities)]),
            f"cumulative protocol violated: CAV{use_cav} got {modalities}; "
            "a slot may be out of communication range. Refusing to reorder/filter/score this population.")
