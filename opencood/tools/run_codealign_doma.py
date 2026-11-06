"""Explicit offline preparation and one-stage-at-a-time server execution.

Nothing starts training/inference on import, prepare, audit or summarize.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import sys
import shutil
import uuid
import traceback
from pathlib import Path

from opencood.tools.codealign_protocol import (
    ROOT, MODALITIES, STAGES, assignments, config_root, digest, evaluation_config,
    load_config, prepare, require, sha256, write_json,
    validate_prefix,
)


class Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    def write(self, text):
        self.console.write(text)
        self.stream.write(text)
        self.stream.flush()

    def flush(self):
        self.console.flush()
        self.stream.flush()


@contextlib.contextmanager
def stage_log(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            try:
                yield
            except BaseException:
                traceback.print_exc(file=stream)
                raise


def runtime_config(experiment, stage):
    config = load_config(Path(experiment) / stage / "config.yaml")
    require(config["codealign_protocol"]["stage"] == stage, "stage directory/config mismatch")
    dataset = config["codealign_protocol"]["dataset"]
    assignments(dataset, config=config)
    require(config["model"]["args"]["lidar_range"] == config["cav_lidar_range"], "model/canvas mismatch")
    from opencood.hypes_yaml.yaml_utils import load_general_params
    return config, load_general_params(copy.deepcopy(config))


def device_for_run():
    import torch
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1,
            "This protocol runner supports one GPU per stage; use independent stages on two GPUs. DDP is not validated.")
    require(torch.cuda.is_available(), "SERVER_REQUIRED: CUDA/spconv/LSS dependencies required")
    # Physical selection belongs to CUDA_VISIBLE_DEVICES; all model code uses
    # the single visible logical device, never a physical GPU number.
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def make_loader(dataset, config, generator, train=False, evaluation=False):
    from torch.utils.data import DataLoader
    from opencood.tools.codealign_state import seed_worker
    options = config["loader"]
    workers = options["num_workers"]
    require(isinstance(workers, int) and workers >= 0, "invalid worker count")
    require(not options.get("persistent_workers", False), "epoch-boundary exact resume requires fresh workers")
    kwargs = dict(batch_size=1 if evaluation else config["train_params"]["batch_size"],
                  num_workers=workers, pin_memory=options["pin_memory"], shuffle=train,
                  drop_last=train, generator=generator, worker_init_fn=seed_worker,
                  collate_fn=dataset.collate_batch_test if evaluation else dataset.collate_batch_train)
    if workers:
        kwargs["prefetch_factor"] = options["prefetch_factor"]
        kwargs["persistent_workers"] = False
    return DataLoader(dataset, **kwargs)


def train_stage(experiment, stage, resume):
    require(stage != "final_infer", "use assemble for final stage")
    directory = Path(experiment) / stage
    require(not (directory / "complete.json").exists(), "stage already complete; preserve it and use a new experiment")
    require(resume or not any(directory.glob("*.pth")), "existing checkpoint: request --resume or use a new experiment")
    config, hypes = runtime_config(experiment, stage)
    history = directory / "run_history" / uuid.uuid4().hex
    history.mkdir(parents=True, exist_ok=False)
    # Record every attempted launch, including dependency/setup failures.
    write_json(history / "run.json", {"resume": resume, "config": config})
    if resume:
        for filename in ("latest.pth", "best.pth"):
            if (directory / filename).is_file():
                shutil.copy2(directory / filename, history / filename)
    with stage_log(history / "run.log"):
        try:
            _train_stage(experiment, stage, resume, config, hypes, history)
        except BaseException as exc:
            write_json(history / "outcome.json", {"status": "failed", "error": str(exc),
                                                  "traceback": traceback.format_exc()})
            raise
        else:
            write_json(history / "outcome.json", {"status": "complete"})


def population_members(config, population):
    """Record scene and member identities in their actual input order."""
    assigned = assignments(config["codealign_protocol"]["dataset"], config=config)["training"]
    scenes = {key[0] for keys in population.values() for key in keys}
    require(scenes <= set(assigned), "sample scene missing from assignment")
    return [[scene, list(map(list, members.items()))] for scene, members in assigned.items() if scene in scenes]


def _train_stage(experiment, stage, resume, config, hypes, history):
    import torch
    from tensorboardX import SummaryWriter
    from opencood.data_utils.datasets.codealign_protocol_dataset import build_protocol_dataset, sample_keys
    from opencood.tools import train_utils
    from opencood.tools.codealign_state import (
        atomic_torch_save, initialize_model, parent_checkpoints, provenance,
        restore_training, rng_state, seed_everything, create_scheduler, verify_checkpoint_identity,
    )
    directory = Path(experiment) / stage
    device = device_for_run()
    seed_everything(config["seed"], config["reproducibility"])
    parents, parent_records = parent_checkpoints(experiment, config)
    identity = provenance(config, parent_records)
    model = train_utils.create_model(hypes).to(device)
    owners = initialize_model(model, stage, parents)
    del parents
    # Freeze BEFORE constructing optimizer; exceptions are not swallowed.
    model.model_train_init()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if stage.startswith("fcf/"):
        require(trainable and all(n.startswith("translators.") for n in trainable), "FCF freeze failure")
    if stage.startswith("code_space/"):
        require(trainable and all(n.startswith(("aligners.", "codebooks.")) for n in trainable), "code-space freeze failure")
    method = config["optimizer"]
    optimizer = getattr(torch.optim, method["core_method"])(
        [p for p in model.parameters() if p.requires_grad], lr=method["lr"], **method.get("args", {}))
    scheduler = create_scheduler(config, optimizer)
    print("Scheduler cadence: one step per epoch; ReduceLROnPlateau steps on validation epochs.")
    initialized_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if any(rate != float(method["lr"]) for rate in initialized_rates):
        print(f"Scheduler initialized learning rates to {initialized_rates}; configured optimizer lr was {method['lr']}.")
    criterion = train_utils.create_loss(config).to(device)
    generators = {name: torch.Generator().manual_seed(config["seed"] + offset)
                  for name, offset in (("train", 0), ("validate", 1))}
    train_dataset = build_protocol_dataset(hypes, train=True)
    val_dataset = build_protocol_dataset(hypes, train=False)
    population = {"train": sample_keys(train_dataset), "validate": sample_keys(val_dataset)}
    require(not (set(map(tuple, population["train"])) & set(map(tuple, population["validate"]))), "train/validation sample leakage")
    data_hash = digest(population)
    members = population_members(config, population)
    audit = {"provenance": identity, "data_sha256": data_hash,
               "sample_members": members,
               "sample_counts": {k: len(v) for k, v in population.items()},
               "weight_owners": owners, "trainable": trainable, "logical_device": str(device),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "resume_granularity": "completed epoch; interrupted epoch reruns"}
    train_loader = make_loader(train_dataset, config, generators["train"], train=True)
    val_loader = make_loader(val_dataset, config, generators["validate"])
    require(len(train_loader) > 0, "training shard is smaller than configured batch size")
    start, best, best_epoch = 0, float("inf"), 0
    if resume:
        state = restore_training(directory / "latest.pth", model, optimizer, scheduler, generators, identity,
                                 config=config, experiment=experiment)
        require(state["data_sha256"] == data_hash, "data population changed on resume")
        if "sample_members" in state:
            require(state["sample_members"] == members, "sample members changed on resume")
        else:
            print("Legacy checkpoint: sample identities verified; historical member list was not recorded.")
        start, best, best_epoch = state["epoch"], state["best_val"], state["best_epoch"]
    audit_path = directory / "input_audit.json"
    if audit_path.exists():
        previous_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        verify_checkpoint_identity(previous_audit["provenance"], identity)
        for field in ("data_sha256", "sample_counts", "weight_owners", "trainable"):
            require(previous_audit[field] == audit[field], f"input audit changed: {field}")
        if "sample_members" in previous_audit:
            require(previous_audit["sample_members"] == members, "input audit sample members changed")
    if not audit_path.exists():
        write_json(audit_path, audit)
    write_json(history / "input_audit.json", audit)
    write_json(history / "run.json", {"resume": resume, "config": config,
               "previous_config": state.get("config") if resume else None,
               "resumed_epoch": start, "provenance": identity,
               "effective_learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
               "optimizer_type": type(optimizer).__name__, "scheduler_type": type(scheduler).__name__,
               "scheduler_last_epoch": scheduler.last_epoch,
               "restore_policy": state.get("restore_policy") if resume else None})
    writer = SummaryWriter(str(history / "tensorboard"))
    try:
        for epoch in range(start, config["train_params"]["epoches"]):
            # Rebuild the member ordering and worker streams deterministically
            # for this epoch, including after an interrupted process restart.
            seed_everything(config["seed"] + epoch * 1009, config["reproducibility"])
            generators["train"].manual_seed(config["seed"] + epoch * 1009)
            generators["validate"].manual_seed(config["seed"] + epoch * 1009 + 1)
            train_dataset.reinitialize()
            model.train()
            train_sum, train_samples = 0.0, 0
            epoch_rates = [float(group["lr"]) for group in optimizer.param_groups]
            for group_index, rate in enumerate(epoch_rates):
                writer.add_scalar(f"train/learning_rate_{group_index}", rate, epoch + 1)
            for index, batch in enumerate(train_loader):
                require(batch is not None, "invalid training sample; refusing silent skip")
                batch = train_utils.to_device(batch, device)
                batch["ego"]["epoch"] = epoch
                optimizer.zero_grad(set_to_none=True)
                output = model(batch["ego"])
                loss = criterion(output, batch["ego"]["label_dict"])
                require(bool(torch.isfinite(loss)), "nonfinite training loss")
                loss.backward()
                optimizer.step()
                count = int(batch["ego"]["record_len"].numel())
                train_sum += loss.item() * count
                train_samples += count
                criterion.logging(epoch, index, len(train_loader), writer)
            writer.add_scalar("train/loss", train_sum / train_samples, epoch + 1)
            validation = None
            improved = False
            if (epoch + 1) % config["train_params"]["eval_freq"] == 0 or epoch + 1 == config["train_params"]["epoches"]:
                model.eval()
                total, samples = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        require(batch is not None, "invalid validation sample")
                        batch = train_utils.to_device(batch, device)
                        batch["ego"]["epoch"] = epoch
                        loss = criterion(model(batch["ego"]), batch["ego"]["label_dict"])
                        count = int(batch["ego"]["record_len"].numel())
                        total += loss.item() * count
                        samples += count
                require(samples > 0, "empty validation population")
                validation = total / samples
                require(math.isfinite(validation), "nonfinite validation loss")
                writer.add_scalar("validate/loss", validation, epoch + 1)
                if validation < best:
                    best, best_epoch, improved = validation, epoch + 1, True
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if validation is not None:
                    scheduler.step(validation)
            else:
                scheduler.step()
            state = {"format": "codealign_protocol_v2", "kind": "training", "epoch": epoch + 1,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "best_val": best, "best_epoch": best_epoch, "validation_loss": validation,
                     "rng": rng_state(generators), "provenance": identity, "data_sha256": data_hash,
                     "config": config, "sample_members": members}
            if improved:
                atomic_torch_save(directory / "best.pth", state)
            atomic_torch_save(directory / "latest.pth", state)
            if (epoch + 1) % config["train_params"]["save_freq"] == 0:
                epoch_checkpoint = directory / f"epoch_{epoch+1:04d}.pth"
                require(not epoch_checkpoint.exists(), f"refusing to overwrite checkpoint: {epoch_checkpoint}")
                atomic_torch_save(epoch_checkpoint, state)
            print(json.dumps({"epoch": epoch + 1, "train_loss": train_sum/train_samples,
                              "validation_loss": validation, "best_epoch": best_epoch,
                              "learning_rates": epoch_rates}), flush=True)
        require((directory / "best.pth").exists(), "no validated checkpoint")
        write_json(directory / "complete.json", {"checkpoint_sha256": sha256(directory / "best.pth"),
                   "best_epoch": best_epoch, "best_validation_loss": best, "provenance": identity})
    finally:
        writer.close()


def assemble(experiment):
    from opencood.tools import train_utils
    from opencood.tools.codealign_state import atomic_torch_save, initialize_model, parent_checkpoints, provenance
    config, hypes = runtime_config(experiment, "final_infer")
    directory = Path(experiment) / "final_infer"
    require(not (directory / "best.pth").exists(), "preserve existing final checkpoint; use a new experiment")
    device_for_run()
    parents, records = parent_checkpoints(experiment, config)
    model = train_utils.create_model(hypes, train_flag=False)
    owners = initialize_model(model, "final_infer", parents)
    identity = provenance(config, records)
    atomic_torch_save(directory / "best.pth", {"format": "codealign_protocol_v2", "kind": "assembly", "epoch": 0,
                      "model": model.state_dict(), "provenance": identity})
    write_json(directory / "assembly.json", {"provenance": identity, "owners": owners,
                                             "checkpoint_sha256": sha256(directory / "best.pth")})


def load_assembly(directory, identity):
    from opencood.tools.codealign_state import load_checkpoint, verify_checkpoint_identity

    directory = Path(directory)
    record = directory / "assembly.json"
    require(record.is_file(), "missing assembly completion record")
    assembly = json.loads(record.read_text(encoding="utf-8"))
    require(assembly["checkpoint_sha256"] == sha256(directory / "best.pth"), "assembled checkpoint changed")
    verify_checkpoint_identity(assembly["provenance"], identity)
    checkpoint = load_checkpoint(directory / "best.pth")
    require(checkpoint["kind"] == "assembly", "final checkpoint is not an assembly")
    verify_checkpoint_identity(checkpoint["provenance"], identity)
    return checkpoint


def infer(experiment, use_cav):
    import numpy as np
    import torch
    from opencood.data_utils.datasets.codealign_protocol_dataset import build_protocol_dataset, sample_keys
    from opencood.tools import train_utils
    from opencood.tools.codealign_state import (parent_checkpoints, provenance,
                                               seed_everything, validate_model_state)
    from opencood.utils import eval_utils
    config, hypes = runtime_config(experiment, "final_infer")
    directory = Path(experiment) / "final_infer"
    out = directory / f"CAV{use_cav}"
    require(not (out / "result.json").exists(), "preserve completed inference result")
    device = device_for_run()
    seed_everything(config["seed"], config["reproducibility"])
    parents, records = parent_checkpoints(experiment, config)
    del parents
    identity = provenance(config, records)
    checkpoint = load_assembly(directory, identity)
    model = train_utils.create_model(hypes, train_flag=False)
    validate_model_state(model, checkpoint["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    hypes = evaluation_config(hypes, use_cav)
    dataset = build_protocol_dataset(hypes, train=False, evaluation=True)
    keys = sample_keys(dataset)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = make_loader(dataset, config, generator, evaluation=True)
    result_stat = {iou: {"fp": [], "tp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}
    ledger, gt_hash = [], hashlib.sha256()
    seen = set()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            require(batch is not None, f"invalid test sample: {keys[index]}; no scenario filtering allowed")
            modalities = batch["ego"]["agent_modality_list"]
            validate_prefix(modalities, use_cav)
            # Frames with fewer vehicles remain in all four populations. A
            # skipped middle modality is an error, never silently substituted.
            seen.update(modalities)
            batch = train_utils.to_device(batch, device)
            output = {"ego": model(batch["ego"])}
            boxes, scores, gt = dataset.post_process(batch, output)
            canonical = gt.detach().cpu().numpy().reshape(len(gt), 24) if len(gt) else np.empty((0, 24))
            if len(canonical):
                canonical = canonical[np.lexsort(canonical.T[::-1])]
            gt_hash.update(json.dumps(keys[index]).encode())
            gt_hash.update(np.asarray(canonical, dtype="<f4").tobytes())
            ledger.append({"key": keys[index], "modalities": modalities, "gt_count": len(gt)})
            for iou in result_stat:
                eval_utils.caluclate_tp_fp(boxes, scores, gt, result_stat, iou)
    require(len(ledger) == len(keys), "test population was truncated")
    require(model.path_counts["codebook_m1"] > 0, "ego Codebook path was bypassed")
    for m in seen - {"m1"}:
        require(model.path_counts[f"fcf_{m}_to_m1"] > 0, f"FCF path was bypassed: {m}")
    out.mkdir(parents=True, exist_ok=True)
    ap = {f"ap{int(iou*100)}": float(eval_utils.calculate_ap(result_stat, iou)[0]) for iou in result_stat}
    require(all(math.isfinite(v) for v in ap.values()), "invalid AP result")
    result = {"dataset": config["codealign_protocol"]["dataset"], "use_cav": use_cav, **ap,
              "samples": len(keys), "sample_sha256": digest(keys), "gt_sha256": gt_hash.hexdigest(),
              "checkpoint_sha256": sha256(directory / "best.pth"),
              "assignment_members": [[scene, list(map(list, members.items()))] for scene, members in
                                     assignments(config["codealign_protocol"]["dataset"], config=config)["evaluation"].items()],
              "range": config["cav_lidar_range"], "comm_range": config["comm_range"],
              "path_counts": dict(model.path_counts), "provenance": identity}
    write_json(out / "raw_statistics.json", {str(k): v for k, v in result_stat.items()})
    write_json(out / "sample_ledger.json", ledger)
    write_json(out / "result.json", result)
    print("CAV%d AP30=%.6f AP50=%.6f AP70=%.6f" % (use_cav, ap["ap30"], ap["ap50"], ap["ap70"]))


def summarize(experiment):
    directory = Path(experiment) / "final_infer"
    rows = [json.loads((directory / f"CAV{n}/result.json").read_text()) for n in (1, 2, 3, 4)]
    for field in ("dataset", "samples", "sample_sha256", "gt_sha256", "checkpoint_sha256",
                  "range", "comm_range"):
        require(all(row[field] == rows[0][field] for row in rows), f"CAV populations/protocol differ: {field}")
    recorded_members = [row["assignment_members"] for row in rows if "assignment_members" in row]
    if recorded_members:
        require(all(value == recorded_members[0] for value in recorded_members), "CAV assignment members differ")
    ledgers = [json.loads((directory / f"CAV{n}/sample_ledger.json").read_text()) for n in (1, 2, 3, 4)]
    require(all(len(ledger) == rows[0]["samples"] for ledger in ledgers), "sample ledger length differs")
    for index in range(rows[0]["samples"]):
        largest = ledgers[-1][index]["modalities"]
        for n in (1, 2, 3, 4):
            require(ledgers[n-1][index]["key"] == ledgers[-1][index]["key"], "sample ledger identity/order differs")
            require(ledgers[n-1][index]["modalities"] == largest[:n], "members were reordered across CAV runs")
    write_json(directory / "ap_summary.json", rows)
    with (directory / "ap_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "use_cav", "ap30", "ap50", "ap70", "samples"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] if not key.startswith("ap") else f"{row[key]:.8f}" for key in writer.fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "assemble", "infer", "summarize"))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("opv2v", "v2xset"))
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-cav", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "prepare":
        require(args.dataset, "prepare requires --dataset")
        prepare(args.experiment_root, args.dataset)
        print("Prepared reviewed YAML copies; no training or inference started.")
    elif args.action == "train":
        require(args.stage, "train requires --stage")
        with stage_log(args.experiment_root / args.stage / "train.log"):
            train_stage(args.experiment_root, args.stage, args.resume)
    elif args.action == "assemble":
        with stage_log(args.experiment_root / "final_infer/assembly.log"):
            assemble(args.experiment_root)
    elif args.action == "infer":
        require(args.use_cav, "infer requires --use-cav")
        with stage_log(args.experiment_root / f"final_infer/CAV{args.use_cav}/inference.log"):
            infer(args.experiment_root, args.use_cav)
    else:
        summarize(args.experiment_root)


if __name__ == "__main__":
    main()
