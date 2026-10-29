"""CPU contract tests; full sensor-model CUDA tests are deliberately separate."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from opencood.models.codealign_doma_protocol import (
    center_canvas, decode_logits, make_codebook, quantize_feature,
)
from opencood.tools.codealign_protocol import (
    MODALITIES, STAGES, assignments, config_root, evaluation_config, load_config,
    prepare, scene_owner, validate_config,
    validate_prefix,
)
from opencood.tools.codealign_state import (
    atomic_torch_save, initialize_model, restore_training, rng_state, seed_everything,
)
from opencood.tools.run_codealign_doma import summarize
from opencood.utils.codealign_data import ProtocolAdaptor


class ProtocolTests(unittest.TestCase):
    def test_all_configs_and_negative_sensor_contracts(self):
        for dataset in ("opv2v", "v2xset"):
            for stage in STAGES:
                config = load_config(config_root(dataset) / stage / "config.yaml")
                wrong = copy.deepcopy(config)
                wrong["fusion"]["dataset"] = "opv2v" if dataset == "v2xset" else "v2xset"
                with self.assertRaises(ValueError):
                    validate_config(wrong)
            wrong = load_config(config_root(dataset) / "final_infer/config.yaml")
            wrong["model"]["args"]["modalities"]["m2"]["encoder_args"]["camera_encoder"] = "Resnet101"
            with self.assertRaises(ValueError):
                validate_config(wrong)

    def test_isolated_scenes_and_assignment_order(self):
        for dataset in ("opv2v", "v2xset"):
            value = assignments(dataset)
            groups = [{s for s in value["training"] if scene_owner(s, value["training"]) == m} for m in MODALITIES]
            self.assertEqual(len(set.union(*groups)), len(value["training"]))
            for first in range(4):
                for second in range(first+1, 4):
                    self.assertFalse(groups[first] & groups[second])
            final = load_config(config_root(dataset) / "final_infer/config.yaml")
            for n in range(1, 5):
                cfg = evaluation_config(final, n)
                self.assertEqual(cfg["use_cav"], n)
                self.assertEqual(cfg["validate_dir"], cfg["test_dir"])
                self.assertEqual(cfg["heter"]["mapping_dict"], dict(zip(MODALITIES, MODALITIES)))
        validate_prefix(["m1", "m2"], 4)  # A real frame with only two vehicles.
        for members in (["m2"], ["m1", "m3"], ["m1", "m2", "m3", "m1"]):
            with self.assertRaises(ValueError):
                validate_prefix(members, 4)

    def test_v2xset_does_not_rewrite_pointcloud(self):
        for modality in MODALITIES:
            cfg = load_config(config_root("v2xset") / "final_infer/config.yaml")
            cfg["modality_assignment"] = {"scene": {"1": modality}}
            adaptor = ProtocolAdaptor(cfg, False)
            for path in ("dataset/V2XSET/test/scene/1/000001.pcd", "D:/V2XSET/train/scene/1/000001.pcd"):
                self.assertEqual(adaptor.switch_lidar_channels(modality, path), path)
        cfg = load_config(config_root("opv2v") / "final_infer/config.yaml")
        cfg["modality_assignment"] = {"scene": {"1": "m3"}}
        out = ProtocolAdaptor(cfg, False).switch_lidar_channels("m3", "dataset/OPV2V/test/scene/1/000001.pcd")
        self.assertEqual(Path(out), Path("dataset/OPV2V_Hetero/test/scene/1/000001_32.pcd"))
        self.assertEqual(ProtocolAdaptor(cfg, False).reorder_cav_list(["5", "-1"], "scene"), ["5", "-1"])

    def test_prepare_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepare(tmp, "v2xset")
            prepare(tmp, "v2xset")
            path = Path(tmp) / "single/m1/config.yaml"
            path.write_text("reviewed: true")
            with self.assertRaises(ValueError):
                prepare(tmp, "v2xset")
            self.assertEqual(path.read_text(), "reviewed: true")

    def test_metric_canvas_preserves_scale_and_rejects_bad_shape(self):
        feature = torch.randn(1, 64, 120, 240)
        result = center_canvas(feature, [-96,-48,-3,96,48,1], [-102.4,-51.2,-3,102.4,51.2,1])
        self.assertEqual(tuple(result.shape), (1,64,128,256))
        self.assertTrue(torch.equal(result[...,4:124,8:248], feature))
        self.assertEqual(result[..., :4, :].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(center_canvas(result, [-102.4,-51.2,-3,102.4,51.2,1], [-96,-48,-3,96,48,1]), feature))
        with self.assertRaises(ValueError):
            center_canvas(feature[:,:,:119], [-96,-48,-3,96,48,1], [-96,-48,-3,96,48,1])

    def test_real_native_codebook_and_fcf_gradient(self):
        seed_everything(303)
        codebook = make_codebook({"seg_num": 1, "dict_size": 16, "r": 1})
        feature = torch.randn(2,64,4,6)
        restored, codes, reconstruction = quantize_feature(codebook, feature)
        self.assertEqual(restored.shape, feature.shape)
        self.assertTrue(reconstruction.isfinite().item())
        restored.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in codebook.parameters()))
        codebook.zero_grad(set_to_none=True)
        codebook.requires_grad_(False)
        logits = torch.randn(2,16,4,6, requires_grad=True)
        decoded = decode_logits(codebook, logits, stochastic=True)
        decoded.square().sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in codebook.parameters()))
        first = decode_logits(codebook, logits.detach(), stochastic=False)
        second = decode_logits(codebook, logits.detach(), stochastic=False)
        self.assertTrue(torch.equal(first, second))
        # Verify hard FCF decoding against direct embedding lookup and the
        # native reconstruction heads. Upstream decode() has an unused _ix bug;
        # the supported CodeAlign path is get_feature(one_hot), used above.
        indices = logits.detach().permute(0,2,3,1).reshape(-1,1,16).argmax(-1)
        embedding = codebook.Codebooks[0][0][indices[:, 0]]
        decoder = codebook._decoders[0]
        expected = decoder._restoreHead(decoder._dequantizationHead(embedding))
        expected = expected.reshape(2,4,6,64).permute(0,3,1,2)
        self.assertTrue(torch.allclose(first, expected))

    def test_complete_resume_restores_optimizer_scheduler_and_rng(self):
        seed_everything(303)
        model = nn.Linear(3,1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,3], gamma=0.1)
        generators = {"train": torch.Generator().manual_seed(303)}
        model(torch.ones(2,3)).square().sum().backward()
        optimizer.step()
        scheduler.step()
        identity = {"dataset": "v2xset", "stage": "single/m1"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"latest.pth"
            atomic_torch_save(path, {"format": "codealign_protocol_v2", "kind": "training", "model": model.state_dict(),
                                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                     "rng": rng_state(generators), "provenance": identity})
            expected_rng = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generators["train"]))
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(2,3)).square().sum().backward()
            optimizer.step()
            expected_weight = model.weight.detach().clone()
            restore_training(path, model, optimizer, scheduler, generators, identity)
            actual_rng = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generators["train"]))
            self.assertEqual(actual_rng[0:2], expected_rng[0:2])
            self.assertTrue(torch.equal(actual_rng[2], expected_rng[2]))
            self.assertTrue(torch.equal(actual_rng[3], expected_rng[3]))
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(2,3)).square().sum().backward()
            optimizer.step()
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertEqual(scheduler.last_epoch, 1)
            with self.assertRaises(ValueError):
                restore_training(path, model, optimizer, scheduler, generators, {"dataset": "opv2v"})

    def test_partial_parent_cannot_initialize_frozen_weights(self):
        model = nn.Linear(3,1)
        with self.assertRaises(ValueError):
            initialize_model(model, "code_space/m1", {"single/m1": {"weight": torch.zeros(1,3)}})

    def test_final_tensor_owners_and_frozen_translator_parents(self):
        model = nn.Module()
        for family in ("encoders", "backbones", "aligners", "codebooks"):
            setattr(model, family, nn.ModuleDict({m: nn.Linear(1,1) for m in MODALITIES}))
        model.translators = nn.ModuleDict({m: nn.Linear(1,1) for m in MODALITIES[1:]})
        for name in ("fusion_backbone", "shrink_conv", "cls_head", "reg_head", "dir_head"):
            setattr(model, name, nn.Linear(1,1))
        states = {}
        for i, m in enumerate(MODALITIES, 1):
            states[f"code_space/{m}"] = {k: torch.full_like(v, i) for k, v in model.state_dict().items()
                                          if (k.startswith(tuple(f"{f}.{m}." for f in ("encoders", "backbones", "aligners", "codebooks")))
                                              or not k.startswith(("encoders.", "backbones.", "aligners.", "codebooks.", "translators.")))}
        for i, m in enumerate(MODALITIES[1:], 2):
            state = {k: v.clone() for k, v in states[f"code_space/{m}"].items()
                     if k.startswith(("encoders.", "backbones.", "aligners."))}
            state.update({k: v.clone() for k, v in states["code_space/m1"].items()
                          if not k.startswith(("encoders.", "backbones.", "aligners."))})
            state.update({k: torch.full_like(v, i+10) for k, v in model.state_dict().items() if k.startswith(f"translators.{m}.")})
            states[f"fcf/{m}_to_m1"] = state
        initialize_model(model, "final_infer", states)
        self.assertEqual(model.encoders["m4"].weight.item(), 4)
        self.assertEqual(model.fusion_backbone.weight.item(), 1)
        self.assertEqual(model.translators["m3"].weight.item(), 13)
        states["fcf/m3_to_m1"]["encoders.m3.weight"] += 1
        with self.assertRaises(ValueError):
            initialize_model(model, "final_infer", states)

    def test_freezing_covers_batchnorm_mode_and_parameter_scope(self):
        from opencood.models.codealign_doma_protocol import CodealignDomaProtocol
        for fcf in (False, True):
            model = CodealignDomaProtocol.__new__(CodealignDomaProtocol)
            nn.Module.__init__(model)
            model.is_single, model.is_fcf, model.is_final = False, fcf, False
            for name in ("encoders", "backbones", "aligners", "codebooks", "translators"):
                setattr(model, name, nn.ModuleDict({"m2": nn.BatchNorm1d(2)}))
            for name in ("fusion_backbone", "shrink_conv", "cls_head", "reg_head", "dir_head"):
                setattr(model, name, nn.BatchNorm1d(2))
            model.train()
            self.assertFalse(model.encoders.training)
            self.assertFalse(model.fusion_backbone.training)
            prefixes = ("translators.",) if fcf else ("aligners.", "codebooks.")
            self.assertTrue(all(p.requires_grad == name.startswith(prefixes) for name, p in model.named_parameters()))
            model.eval()
            self.assertFalse(any(m.training for m in model.modules()))

    def test_summary_refuses_different_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"final_infer"
            for n in range(1,5):
                path = directory/f"CAV{n}"
                path.mkdir(parents=True)
                (path/"result.json").write_text(json.dumps({"dataset": "v2xset", "samples": n}))
            with self.assertRaises(ValueError):
                summarize(tmp)


if __name__ == "__main__":
    unittest.main()
