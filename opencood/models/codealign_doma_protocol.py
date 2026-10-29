"""Isolated protocol assembly of CodeAlign's native modules.

No DOMA module or weight is imported. Feature interfaces use a metric 0.8 m
grid and parameter-free center crop/pad; representation learning remains
native Aligner + UMGMQuantizer + Translator + PyramidFusion.
"""
import copy
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.codebook import UMGMQuantizer
from opencood.utils.codebook_utils import gumbelSoftmax


def make_codebook(spec):
    channel = 64
    return UMGMQuantizer(channel, spec["seg_num"], [spec["dict_size"]] * spec["r"], 0.0,
                         {name: (lambda: nn.Linear(channel, channel)) for name in (
                             "latentStageEncoder", "quantizationHead", "latentHead",
                             "restoreHead", "dequantizationHead", "sideHead")})


def metric_shape(bounds, resolution=0.8):
    result = [(bounds[4] - bounds[1]) / resolution, (bounds[3] - bounds[0]) / resolution]
    if any(abs(v - round(v)) > 1e-5 for v in result):
        raise ValueError(f"range does not fit feature grid: {bounds}")
    return tuple(round(v) for v in result)


def center_canvas(feature, source_range, target_range, resolution=0.8):
    if any(abs(source_range[i] + source_range[i + 3]) > 1e-5 for i in (0, 1)) or any(
            abs(target_range[i] + target_range[i + 3]) > 1e-5 for i in (0, 1)):
        raise ValueError("only centered DOMA ranges are supported")
    expected = metric_shape(source_range, resolution)
    if feature.shape[-2:] != expected or feature.shape[1] != 64:
        raise ValueError(f"BEV interface mismatch: {tuple(feature.shape)} != (N,64,{expected})")
    h, w = metric_shape(target_range, resolution)
    # Match torchvision CenterCrop's rounding and zero padding used upstream.
    sh, sw = feature.shape[-2:]
    feature = F.pad(feature, (max(0, (w-sw)//2), max(0, w-sw-(w-sw)//2),
                              max(0, (h-sh)//2), max(0, h-sh-(h-sh)//2)))
    sh, sw = feature.shape[-2:]
    top, left = round((sh-h)/2), round((sw-w)/2)
    return feature[..., top:top+h, left:left+w]


def quantize_feature(codebook, feature):
    n, c, h, w = feature.shape
    decoded, codes, _, reconstruction = codebook(feature.permute(0, 2, 3, 1).reshape(-1, c))
    return decoded.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous(), codes, reconstruction


def decode_logits(codebook, logits, stochastic):
    n, k, h, w = logits.shape
    if k != 16:
        raise ValueError("FCF output must contain 16 code logits")
    flat = logits.permute(0, 2, 3, 1).reshape(-1, 1, k)
    if stochastic:
        assignment = gumbelSoftmax(flat, 1.0, True)
    else:
        assignment = F.one_hot(flat.argmax(-1), num_classes=k).to(logits.dtype)
    # Decode through the ORIGINAL frozen target codebook. Do not detach: the
    # detection gradient must reach the translator through its frozen decoder.
    decoded = codebook.get_feature([assignment])
    return decoded.reshape(n, h, w, 64).permute(0, 3, 1, 2).contiguous()


class CodealignDomaProtocol(nn.Module):
    def __init__(self, args, train_flag=True):
        super().__init__()
        from opencood.models import heter_encoders
        from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
        from opencood.models.sub_modules.feature_alignnet import AlignNet
        from opencood.models.sub_modules.downsample_conv import DownsampleConv
        from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
        from opencood.models.c2c_module import Translator

        self.args = copy.deepcopy(args)
        self.stage = args["stage"]
        self.is_single = self.stage.startswith("single/")
        self.is_fcf = self.stage.startswith("fcf/")
        self.is_final = self.stage == "final_infer"
        self.modality_name_list = list(args["modalities"])
        self.encoders = nn.ModuleDict()
        self.backbones = nn.ModuleDict()
        self.aligners = nn.ModuleDict()
        self.codebooks = nn.ModuleDict()
        self.translators = nn.ModuleDict()
        self.feature_ranges = {}
        self.cam_crop_info = {}
        for m, spec in args["modalities"].items():
            name = spec["core_method"].replace("_", "").lower()
            encoder_class = next(cls for key, cls in vars(heter_encoders).items() if key.lower() == name)
            self.encoders[m] = encoder_class(copy.deepcopy(spec["encoder_args"]))
            self.backbones[m] = ResNetBEVBackbone(copy.deepcopy(spec["backbone_args"]))
            if spec["sensor_type"] == "camera":
                grid = spec["encoder_args"]["grid_conf"]
                bounds = [grid["xbound"][0], grid["ybound"][0], -3,
                          grid["xbound"][1], grid["ybound"][1], 1]
                self.cam_crop_info[m] = {f"crop_ratio_H_{m}": args["lidar_range"][4] / bounds[4],
                                         f"crop_ratio_W_{m}": args["lidar_range"][3] / bounds[3]}
            else:
                bounds = spec["encoder_args"]["lidar_range"]
            self.feature_ranges[m] = bounds
            if not self.is_single:
                self.aligners[m] = AlignNet(copy.deepcopy(args["aligner"]))
            if self.is_fcf or (self.is_final and m != "m1"):
                self.translators[m] = Translator(copy.deepcopy(args["translator"]))
            if not self.is_single and (not self.is_fcf or m == "m1"):
                self.codebooks[m] = make_codebook(args["codebook"])
        if self.is_fcf:
            self.codebooks["m1"] = make_codebook(args["codebook"])
        self.fusion_backbone = PyramidFusion(copy.deepcopy(args["fusion"]["backbone"]))
        self.shrink_conv = DownsampleConv(copy.deepcopy(args["fusion"]["shrink_head"]))
        self.cls_head = nn.Conv2d(args["in_head"], args["anchor_number"], 1)
        self.reg_head = nn.Conv2d(args["in_head"], 7 * args["anchor_number"], 1)
        self.dir_head = nn.Conv2d(args["in_head"], args["dir_args"]["num_bins"] * args["anchor_number"], 1)
        self.path_counts = Counter()
        self.model_train_init()

    def model_train_init(self):
        if self.is_single:
            return
        self.requires_grad_(False)
        # Freeze module mode as well as gradients, including BN running stats.
        for module in (self.encoders, self.backbones, self.fusion_backbone, self.shrink_conv,
                       self.cls_head, self.reg_head, self.dir_head, self.aligners, self.codebooks):
            module.eval()
        if self.is_final:
            self.translators.eval()
        elif self.is_fcf:
            self.translators.requires_grad_(True)
            self.translators.train(self.training)
        else:
            for module in (self.aligners, self.codebooks):
                module.requires_grad_(True)
                module.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.model_train_init()
        return self

    def forward(self, data_dict):
        from opencood.utils.transformation_utils import normalize_pairwise_tfm

        agents = data_dict["agent_modality_list"]
        if not set(agents) <= set(self.modality_name_list):
            raise ValueError(f"unconfigured input modality: {agents}")
        if not self.is_final and len(set(agents)) != 1:
            raise ValueError("modality isolation violation: mixed training frame")
        if self.is_final and agents[0] != "m1":
            raise ValueError("ordered inference requires m1 ego")
        features, output = {}, {}
        for m in dict.fromkeys(agents):
            feature = self.encoders[m](data_dict, m)
            feature = self.backbones[m]({"spatial_features": feature})["spatial_features_2d"]
            if not self.is_single:
                feature = self.aligners[m](feature)
            feature = center_canvas(feature, self.feature_ranges[m], self.args["lidar_range"],
                                    self.args["feature_resolution"])
            spec = self.args["modalities"][m]
            if spec["encoder_args"].get("depth_supervision", False):
                output[f"depth_items_{m}"] = self.encoders[m].depth_items
            if not self.is_single:
                if self.is_fcf or (self.is_final and m != "m1"):
                    logits = self.translators[m](feature, "m1")
                    if logits.shape[-2:] != feature.shape[-2:]:
                        raise ValueError("translator changed the metric canvas")
                    feature = decode_logits(self.codebooks["m1"], logits, self.is_fcf and self.training)
                    self.path_counts[f"fcf_{m}_to_m1"] += 1
                else:
                    feature, _, reconstruction = quantize_feature(self.codebooks[m], feature)
                    output["codebook_loss"] = reconstruction
                    self.path_counts[f"codebook_{m}"] += 1
            features[m] = feature
        counts = Counter()
        ordered = []
        for m in agents:
            ordered.append(features[m][counts[m]])
            counts[m] += 1
        feature = torch.stack(ordered)
        bounds = self.args["lidar_range"]
        affine = normalize_pairwise_tfm(data_dict["pairwise_t_matrix"],
                                        bounds[4]-bounds[1], bounds[3]-bounds[0], 1)
        fused, occupancy = self.fusion_backbone.forward_collab(
            feature, data_dict["record_len"], affine, agents, self.cam_crop_info)
        fused = self.shrink_conv(fused)
        output.update(cls_preds=self.cls_head(fused), reg_preds=self.reg_head(fused),
                      dir_preds=self.dir_head(fused), pyramid="collab", occ_single_list=occupancy)
        return output
