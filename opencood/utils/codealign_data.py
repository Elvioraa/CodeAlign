"""Point-file and member assignment interfaces; no DL imports at import time."""
import random
from pathlib import Path
from opencood.tools.codealign_protocol import require


def read_protocol_pcd(path):
    """Keep RGB-encoded HEAL intensity; support real XYZI PCD without colors."""
    import numpy as np
    from opencood.utils.pcd_utils import pcd_to_np
    fields = []
    with Path(path).open("rb") as stream:
        for _ in range(100):
            line = stream.readline().decode("ascii", errors="replace").strip()
            if line.upper().startswith("FIELDS "):
                fields = line.split()[1:]
                break
            if line.upper().startswith("DATA "):
                break
    if "rgb" in fields or "rgba" in fields:
        result = pcd_to_np(str(path))
    else:
        from pypcd import pypcd
        require(all(name in fields for name in ("x", "y", "z", "intensity")),
                f"PCD must supply XYZ and intensity/RGB, cannot invent a feature: {path}")
        cloud = pypcd.PointCloud.from_path(str(path))
        result = np.column_stack([cloud.pc_data[name] for name in ("x", "y", "z", "intensity")]).astype(np.float32)
    require(result.ndim == 2 and result.shape[1] == 4 and np.isfinite(result).all(),
            f"invalid XYZI features: {path}")
    return result


class ProtocolAdaptor:
    """No hidden seed reset, inferred sensor ID, or V2XSet beam conversion."""
    def __init__(self, params, train):
        self.params = params
        self.train = train
        self.modality_assignment = params["modality_assignment"]
        self.mapping_dict = params["heter"]["mapping_dict"]
        self.modalities = params["heter"]["modality_setting"]

    def reorder_cav_list(self, cav_list, scenario_name):
        if self.train:
            random.shuffle(cav_list)
        return cav_list

    def reassign_cav_modality(self, modality_name, idx_in_cav_list, scenario_name):
        return self.mapping_dict[modality_name]

    def unmatched_modality(self, cav_modality):
        return cav_modality not in self.modalities

    def switch_lidar_channels(self, cav_modality, lidar_file_path):
        if self.params["fusion"]["dataset"] == "v2xset":
            require(not self.params["heter"].get("lidar_channels_dict"),
                    "V2XSet must use original PCD without beam rewriting")
            return lidar_file_path
        if cav_modality == "m3":
            path = Path(lidar_file_path)
            parts = list(path.parts)
            require("OPV2V" in parts, "OPV2V-H conversion requires an explicit OPV2V root")
            parts[parts.index("OPV2V")] = "OPV2V_Hetero"
            return str(Path(*parts).with_name(path.stem + "_32.pcd"))
        return lidar_file_path


