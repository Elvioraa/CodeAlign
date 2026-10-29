"""Dataset-only protocol selection; reuse CodeAlign labels and preprocessing."""
import copy
import json
import random
from pathlib import Path

from opencood.tools.codealign_protocol import ROOT, MODALITIES, require, scene_owner


from opencood.utils.codealign_data import ProtocolAdaptor, read_protocol_pcd


def build_protocol_dataset(config, train=False, evaluation=False):
    from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
    from opencood.data_utils.datasets.basedataset.v2xset_basedataset import V2XSETBaseDataset
    from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset
    from opencood.data_utils.datasets.heter_infer.intermediate_heter_infer_fusion_dataset import getIntermediateheterinferFusionDataset

    params = copy.deepcopy(config)
    protocol = params["codealign_protocol"]
    assignment = json.loads((ROOT / params["heter"]["assignment_path"]).read_text(encoding="utf-8"))
    if not evaluation:
        # Same disjoint modality scenes for pretrain, code space and FCF. Test
        # frames are never used here; base loader selects train/validate only.
        assignment = {s: c for s, c in assignment.items() if scene_owner(s, assignment) == protocol["source"]}
    assignment = {s: c for s, c in assignment.items() if s not in protocol["camera_excluded_scenes"]}
    params["modality_assignment"] = assignment
    params["val"] = not train and not evaluation
    base = V2XSETBaseDataset if params["fusion"]["dataset"] == "v2xset" else OPV2VBaseDataset
    factory = getIntermediateheterinferFusionDataset if evaluation else getIntermediateheterFusionDataset
    # DOMA's actual ordered entry passes visualize=True. This flag also selects
    # LiDAR-derived GT for ALL modalities, not only point-cloud visualization.
    # Preserve that semantic; False would silently change camera GT and AP.
    dataset = factory(base)(params, evaluation, train)
    require(dataset.len_record and len(dataset) > 0, "empty source shard/split; check assignment and real data")
    if evaluation:
        # The upstream infer dataset historically hardcodes m2/m4 as LiDAR.
        dataset.sensor_type_dict = {m: s["sensor_type"] for m, s in params["heter"]["modality_setting"].items()}
    dataset.protocol_assignment = assignment
    return dataset


def sample_keys(dataset):
    result = []
    for index, scenario in dataset.scenario_database.items():
        ego = next(iter(scenario.values()))
        for timestamp in ego:
            if timestamp == "ego":
                continue
            result.append([Path(dataset.scenario_folders[index]).name, timestamp])
    require(len(result) == len(dataset), "sample ledger length mismatch")
    return result
