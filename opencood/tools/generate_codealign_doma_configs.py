"""Explicit, offline extraction of reference configs; never used by the trainer."""
import argparse
import copy
import json
from pathlib import Path

import yaml

from opencood.tools.codealign_protocol import (
    ROOT, MODALITIES, STAGES, config_root, digest, ordered_hash, sha256, stage_parents,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def generate(reference):
    source_root = ROOT / "docs/protocol_sources"
    snapshot_path = reference / "opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2_R2_TABLE4_EXT/history_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    records = {}
    for dataset in ("opv2v", "v2xset"):
        refs, provenance = {}, {}
        names = {"m1": "stage1/m1_base", "m2": "stage2/m2_alignto_m1_lr0p0012",
                 "m3": "stage2/m3_alignto_m1_lr0p0030", "m4": "stage2/m4_alignto_m1_lr0p0017",
                 "final": "final_infer"}
        v2_names = {"m1": "stage1/m1", "m2": "stage2/m2", "m3": "stage2/m3", "m4": "stage2/m4",
                    "final": "final_infer/m1m2m3m4"}
        for name in (*MODALITIES, "final"):
            if dataset == "opv2v":
                key = f"opencood/logs/DOMA_V2_R2_2GPU_LR1P414_SEED303/{names[name]}/config.yaml"
                text = snapshot[key]
                origin = str(snapshot_path) + "#" + key
            else:
                path = reference / "opencood/hypes_yaml/v2xset/MoreModality/DOMA/V2" / (v2_names[name] + ".yaml")
                text = path.read_text(encoding="utf-8")
                origin = str(path)
            out = source_root / dataset / (name + ".yaml")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            refs[name] = yaml.safe_load(text)
            provenance[name] = {"origin": origin, "copy": out.relative_to(ROOT).as_posix(),
                                "sha256": sha256(out), "parsed_sha256": digest(refs[name]),
                                "status": "saved_run_snapshot" if dataset == "opv2v" else "local_reference_only"}
        base = config_root(dataset)
        assignment_records = {}
        for kind, suffix in (("training", ""), ("evaluation", "_in_order")):
            path = ROOT / f"opencood/modality_assign/{dataset}_4modality{suffix}.json"
            value = json.loads(path.read_text())
            reference_path = reference / path.relative_to(ROOT)
            assignment_records[kind] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                                        "ordered_sha256": ordered_hash(value), "scenarios": len(value),
                                        "equals_local_DOMA": json.loads(reference_path.read_text()) == value}
        final = refs["final"]
        evaluate_range = [-204.8, -102.4, -3, 204.8, 102.4, 1]
        lock = {"schema_version": 2, "dataset": dataset, "codealign_base": "75d5ecd359d36a64a3af3d64c4fe32e72ed344bc",
                "reference_doma_commit": "ef19dd01fbbbe35b82956b37c7b1a5b4bbf1bfd8",
                "reference_server": "/home/fkk/my_data/disk2/code/DOMA",
                "seed": 303, "reference_configs": provenance, "assignments": assignment_records,
                "data_identity": "OPV2V with OPV2V_Hetero 32-beam and depth extensions (OPV2V-H protocol)" if dataset == "opv2v" else "V2XSet original PCD and four RGB cameras",
                "sensor_policy": {"m1": "original_PCD_PointPillar", "m2": "LSS_EfficientNet",
                                  "m3": "OPV2V_Hetero_32_PCD_SECOND" if dataset == "opv2v" else "original_PCD_SECOND",
                                  "m4": "LSS_Resnet101"},
                "sensor_policy_source": "user clarification 2026-10-27; DOMA local configs",
                "splits": {k: final[k] for k in ("root_dir", "validate_dir", "test_dir")},
                "training": {"comm_range": 70, "max_cav": 5,
                             "modality_ranges": {m: refs[m]["cav_lidar_range"] for m in MODALITIES},
                             "isolation": "insertion_order_scene_index_mod_4; disjoint source scenes in every stage",
                             "budget_policy": "CodeAlign native templates; not DOMA/STAMP training budgets",
                             "epochs": {"single": 50, "code_space": 50, "fcf": 30}},
                "evaluation": {"range": evaluate_range, "comm_range": 180, "max_cav": 5,
                               "use_cav": [1, 2, 3, 4], "ego": "m1", "gt": "all_in_range_agents_before_prefix_cutoff",
                               "gt_label_source": "lidar_for_every_modality_via_visualize_true",
                               "iou": [0.3, 0.5, 0.7], "postprocess": final["postprocess"],
                               "range_source": "DOMA inference_doma_heter_in_order.py default/recursive override",
                               "runtime_confirmation": "SERVER_VALIDATION_REQUIRED",
                               "note": "OPV2V saved logs confirm dataset/AP, not historical CLI flags. V2XSet is a local reference, not a server run snapshot."},
                "server_validation_required": ["actual data identity and split/sample counts", "actual DOMA final range/GT/NMS/CLI provenance",
                                               "CUDA/spconv DataLoader and full forward/backward", "training, full-state resume and final AP"],
                "camera_excluded_scenes": [] if dataset == "opv2v" else ["2021_09_09_13_20_58"]}
        dump(base / "protocol_lock.yaml", lock)
        entries = {}
        for stage in STAGES:
            final_stage = stage == "final_infer"
            source = "m1" if final_stage else stage.split("/")[1].split("_")[0]
            template = copy.deepcopy(final if final_stage else refs[source])
            # Remove all method-specific state. Only sensor definitions come from DOMA.
            sensor_models, sensor_data = {}, {}
            for m in (MODALITIES if final_stage else (source,)):
                setting = copy.deepcopy((final if final_stage else refs[m])["model"]["args"][m])
                setting.pop("aligner_args", None)
                setting.pop("model_path", None)
                sensor_models[m] = setting
                sensor_data[m] = copy.deepcopy((final if final_stage else refs[m])["heter"]["modality_setting"][m])
            if final_stage:
                # Reproduce DOMA's recursive inference range override, explicitly in YAML.
                canvas = evaluate_range
                for m in ("m1", "m3"):
                    sensor_models[m]["encoder_args"]["lidar_range"] = list(canvas)
                    sensor_data[m]["preprocess"]["cav_lidar_range"] = list(canvas)
            elif stage.startswith("fcf/"):
                canvas = refs["m1"]["cav_lidar_range"]
            else:
                canvas = refs[source]["cav_lidar_range"]
            anchor = copy.deepcopy(template["postprocess"])
            anchor["anchor_args"]["cav_lidar_range"] = list(canvas)
            anchor["gt_range"] = list(canvas)
            anchor["cav_lidar_range"] = list(canvas)
            anchor["voxel_size"] = [0.4, 0.4, 4]
            # Label stride 2 on the common 0.4 m grid, matching all 0.8 m feature grids.
            preprocess = copy.deepcopy(template["preprocess"])
            preprocess["cav_lidar_range"] = list(canvas)
            preprocess["args"]["voxel_size"] = [0.4, 0.4, 4]
            fusion_args = refs["m1"]["model"]["args"]
            transformer = {"core_method": "convnext", "args": {
                "in_channels": 64, "out_channels": 16, "in_cav_lidar_range": list(canvas),
                "out_cav_lidar_range": list(canvas), "in_feature_shape": [int(round((canvas[4]-canvas[1])/0.8)), int(round((canvas[3]-canvas[0])/0.8))],
                "out_feature_shape": [int(round((canvas[4]-canvas[1])/0.8)), int(round((canvas[3]-canvas[0])/0.8))],
                "submodule_args": {"num_of_blocks": 3, "dim": 64, "modality_map": ["m1"]}}}
            native = ROOT / "opencood/hypes_yaml/opv2v/Heter_group/codebook"
            budget_name = "codebook_pyramid_c2c.yaml" if stage.startswith("fcf/") else (
                "codebook_pyramid_fixenc_aligner.yaml" if stage.startswith("code_space/") else "pyramid.yaml")
            budget = yaml.safe_load((native / budget_name).read_text())
            for key in ("eps", "weight_decay"):
                if key in budget["optimizer"].get("args", {}):
                    budget["optimizer"]["args"][key] = float(budget["optimizer"]["args"][key])
            loss = copy.deepcopy(budget["loss"])
            # Single-modality code space: original detection+pyramid objective; no inter-modal similarity term.
            loss["core_method"] = "point_pillar_pyramid_loss"
            loss["args"].pop("codebook", None)
            config = {"name": f"CodeAlign_{dataset}_{stage.replace('/', '_')}", "seed": 303,
                      **{k: template[k] for k in ("root_dir", "validate_dir", "test_dir")},
                      "yaml_parser": "load_general_params", "train_params": copy.deepcopy(budget["train_params"]),
                      "comm_range": 180 if final_stage else 70,
                      "input_source": template["input_source"], "label_type": template["label_type"],
                      "cav_lidar_range": list(canvas), "add_data_extension": template.get("add_data_extension", []),
                      "heter": {"assignment_path": assignment_records["evaluation" if final_stage else "training"]["path"],
                                "ego_modality": list(MODALITIES) if final_stage else [source],
                                "heter_group": [list(MODALITIES)] if final_stage else [[source]],
                                "mapping_dict": {m: m if final_stage else source for m in MODALITIES},
                                "lidar_channels_dict": {"m3": 32} if dataset == "opv2v" else {},
                                "modality_setting": sensor_data},
                      "fusion": {"core_method": "intermediateheterinfer" if final_stage else "intermediateheter",
                                 "dataset": dataset, "args": {"proj_first": False, "grid_conf": None, "data_aug_conf": None}},
                      "preprocess": preprocess, "postprocess": anchor,
                      "model": {"core_method": "codealign_doma_protocol", "args": {
                          "stage": stage, "dataset": dataset, "modalities": sensor_models, "channels": 64,
                          "lidar_range": list(canvas), "feature_resolution": 0.8,
                          "anchor_number": anchor["anchor_args"]["num"], "dir_args": anchor["dir_args"],
                          "in_head": fusion_args["in_head"], "supervise_single": False,
                          "fusion": {"method": "pyramidfusion", "backbone": fusion_args["fusion_backbone"],
                                     "shrink_head": fusion_args["shrink_header"]},
                          "aligner": {"core_method": "resnet3x3", "args": {"dim": 64, "num_of_blocks": 4}},
                          "codebook": {"seg_num": 1, "dict_size": 16, "r": 1}, "translator": transformer}},
                      "loss": loss, "optimizer": copy.deepcopy(budget["optimizer"]),
                      "lr_scheduler": copy.deepcopy(budget["lr_scheduler"]),
                      "loader": {"num_workers": 4, "prefetch_factor": 2, "pin_memory": True, "persistent_workers": False},
                      "reproducibility": {"deterministic_algorithms": False, "cudnn_deterministic": True, "cudnn_benchmark": False},
                      "codealign_protocol": {"dataset": dataset, "stage": stage, "source": source, "target": "m1", "seed": 303,
                          "scene_partition": "assignment_insertion_index_mod4", "camera_excluded_scenes": lock["camera_excluded_scenes"],
                          "evaluation_assignment": assignment_records["evaluation"]["path"], "budget_source": (native / budget_name).relative_to(ROOT).as_posix(),
                          "reference_sensor_source": provenance["final" if final_stage else source]["copy"]}}
            config["train_params"]["max_cav"] = 5
            if stage.startswith("code_space/"):
                config["train_params"]["epoches"] = 50
            if final_stage:
                config["input_source"] = ["lidar", "camera", "depth"] if dataset == "opv2v" else ["lidar", "camera"]
                config["label_type"] = "lidar"
                config["use_cav"] = 4
            dump(base / stage / "config.yaml", config)
            entries[stage] = {"config": f"{stage}/config.yaml", "parents": stage_parents(stage),
                              "checkpoint": f"{stage}/best.pth", "budget_source": config["codealign_protocol"]["budget_source"]}
        dump(base / "manifest.yaml", {"dataset": dataset, "stages": entries,
                                      "checkpoint_policy": "unique best validation loss; complete state latest.pth for resume"})
        records[dataset] = lock
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-doma", type=Path, required=True)
    generate(parser.parse_args().reference_doma.resolve())
