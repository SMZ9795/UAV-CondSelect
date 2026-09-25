#!/usr/bin/env python3
"""Detector training with exact resumable sampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader

PROJECT = Path(os.environ['UAV_DATA_ROOT'])
RUN = Path(__file__).resolve().parents[1]
UPSTREAM_REPO = Path(__file__).resolve().parents[2] / 'vendor/rtdetr'
UPSTREAM_PYTHON = UPSTREAM_REPO
CHECKPOINT_SHA256 = "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"
MODEL_COMMIT = "a21d516aca15da57e65f35c47659c7535ad2b6b3"
METRIC_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def stable_seed(*parts: object) -> int:
    data = "\0".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


class StatelessRepeatBatchSampler(BatchSampler):
    """Full batches from concatenated deterministic epoch permutations."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int, start_batch: int = 0):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.start_batch = start_batch

    def permutation(self, epoch: int) -> list[int]:
        values = list(range(self.dataset_size))
        random.Random(stable_seed("calibration-uniform-order", self.seed, epoch)).shuffle(values)
        return values

    def batch(self, batch_number: int) -> list[int]:
        position = batch_number * self.batch_size
        result: list[int] = []
        while len(result) < self.batch_size:
            epoch, offset = divmod(position, self.dataset_size)
            take = min(self.batch_size - len(result), self.dataset_size - offset)
            result.extend(self.permutation(epoch)[offset:offset + take])
            position += take
        return result

    def __iter__(self) -> Iterator[list[int]]:
        batch_number = self.start_batch
        while True:
            yield self.batch(batch_number)
            batch_number += 1

    def __len__(self) -> int:
        return 2**31 - 1


class ProbeBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def _batches(self) -> list[list[int]]:
        metas = [self.dataset.coco.imgs[i] for i in self.dataset.ids]
        high = sorted(range(len(metas)), key=lambda i: int(metas[i]["width"]) * int(metas[i]["height"]), reverse=True)
        diverse: list[int] = []
        for task in ("clean", "rain", "snow", "fog", "dust", "dim_low_light"):
            candidates = [i for i, m in enumerate(metas) if m.get("task") == task]
            diverse.extend(candidates[:max(1, self.batch_size // 6)])
        diverse.extend(i for i in high if i not in diverse)
        return [high[:self.batch_size], diverse[:self.batch_size]]

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        while True:
            yield from batches

    def __len__(self) -> int:
        return 2


def load_official_weights(model, path: Path) -> dict[str, Any]:
    if sha256(path) != CHECKPOINT_SHA256:
        raise RuntimeError("initialization SHA-256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = state["ema"]["module"]
    destination = model.state_dict()
    matched = {k: v for k, v in params.items() if k in destination and tuple(v.shape) == tuple(destination[k].shape)}
    mismatches = sorted(k for k, v in params.items() if k in destination and tuple(v.shape) != tuple(destination[k].shape))
    incompatible = model.load_state_dict(matched, strict=False)
    return {"checkpoint": str(path), "sha256": CHECKPOINT_SHA256, "source_state": "ema.module", "matched_keys": len(matched), "shape_mismatch_keys": mismatches, "missing_after_load": sorted(incompatible.missing_keys), "unexpected_after_load": sorted(incompatible.unexpected_keys)}


def worker_seed(worker_id: int) -> None:
    value = (torch.initial_seed() + worker_id) % 2**32
    random.seed(value)
    np.random.seed(value)


def make_loader(base, sampler, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(base.dataset, batch_sampler=sampler, num_workers=base.num_workers, collate_fn=base.collate_fn, pin_memory=base.pin_memory, worker_init_fn=worker_seed, generator=generator, persistent_workers=False)


def schedule_factor(update_number: int, warmup: int, drop: int, gamma: float) -> float:
    value = min(1.0, update_number / warmup) if warmup > 0 else 1.0
    if update_number >= drop:
        value *= gamma
    return value


def optimizer_report(optimizer, model) -> dict[str, Any]:
    names = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for i, group in enumerate(optimizer.param_groups):
        group_names = [names.get(id(p), "<unknown>") for p in group["params"]]
        groups.append({"index": i, "parameter_tensors": len(group_names), "parameters": sum(p.numel() for p in group["params"]), "lr": group["lr"], "weight_decay": group["weight_decay"], "betas": list(group["betas"]), "backbone_parameter_tensors": sum("backbone" in n for n in group_names), "normalization_parameter_tensors": sum(("norm" in n or "bn" in n) for n in group_names)})
    return {"type": type(optimizer).__name__, "groups": groups, "backbone_lr_multiplier": 1.0, "clip_max_norm": 0.1}


def checkpoint_payload(model, ema, optimizer, scaler, runner: dict, identities: dict, protocol: dict) -> dict:
    return {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "runner": runner, "rng": capture_rng(), "identities": identities, "custom_update_schedule": protocol}


def atomic_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def trim_metrics_for_resume(path: Path, runner: dict) -> None:
    if not path.exists():
        return
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    retained = []
    for r in records:
        if r["kind"] == "validation" and int(r["optimizer_update"]) <= int(runner["successful_updates"]):
            retained.append(r)
        elif r["kind"] in {"train", "amp_skipped"} and int(r["microbatch_attempt"]) <= int(runner["microbatches_consumed"]):
            retained.append(r)
    if len(retained) != len(records):
        tail = path.with_name(f"metrics.interrupted_tail_after_update_{runner['successful_updates']}.jsonl")
        tail.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records[len(retained):]))
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in retained))


def evaluate(cfg, model, ema, device) -> dict[str, float]:
    from src.solver.det_engine import evaluate as upstream_evaluate
    module = ema.module
    stats, _ = upstream_evaluate(module, cfg.criterion, cfg.postprocessor, cfg.val_dataloader, cfg.evaluator, device)
    values = stats["coco_eval_bbox"]
    model.train()
    cfg.criterion.train()
    return {k: float(v) for k, v in zip(METRIC_NAMES, values)}


def assert_state_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()
    config = args.config.resolve()
    sys.path.insert(0, str(UPSTREAM_PYTHON))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config))
    protocol = cfg.yaml_cfg["uav_protocol"]
    condition = protocol["condition"]
    output = (args.output_dir or Path(cfg.output_dir)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE.json").exists() and not args.probe:
        raise RuntimeError(f"duplicate launch prevented; completed run exists at {output}")
    seed = int(protocol["sampler_seed"])
    seed_all(seed)
    assert torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    model = cfg.model
    preload = load_official_weights(model, Path(cfg.tuning))
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    optimizer = cfg.optimizer
    ema = cfg.ema.to(device)
    scaler = cfg.scaler
    base_loader = cfg.train_dataloader
    physical_batch = int(base_loader.batch_size)
    accumulation = int(protocol["gradient_accumulation_steps"])
    effective_batch = physical_batch * accumulation
    if accumulation != 1:
        raise RuntimeError("This training protocol requires gradient_accumulation_steps=1")
    total_updates = int(protocol["total_optimizer_updates"])
    validation_interval = int(protocol["validation_interval_updates"])
    warmup = int(protocol["warmup_updates"])
    drop = int(protocol["lr_drop_update"])
    gamma = float(protocol["lr_drop_factor"])
    base_group_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    annotation = Path(cfg.yaml_cfg["train_dataloader"]["dataset"]["ann_file"])
    identities = {"config_sha256": sha256(config), "runner_sha256": sha256(Path(__file__)), "train_annotation_sha256": sha256(annotation), "val_annotation_sha256": None, "initialization_sha256": CHECKPOINT_SHA256, "model_commit": MODEL_COMMIT}
    resolved = {"yaml": cfg.yaml_cfg, "identities": identities, "optimizer": optimizer_report(optimizer, model), "normalization": {"PResNet.freeze_norm": cfg.yaml_cfg["PResNet"]["freeze_norm"], "PResNet.freeze_at": cfg.yaml_cfg["PResNet"]["freeze_at"], "note": "Physical batch determines BatchNorm statistics; accumulation is not equivalent."}, "augmentation": cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"], "ema": cfg.yaml_cfg["ema"], "evaluation_weights": "EMA", "postprocessor": cfg.yaml_cfg["RTDETRPostProcessor"]}
    atomic_json(output / "resolved_config.json", resolved)
    atomic_json(output / "pretrained_load_report.json", preload)

    assert physical_batch == 16 and accumulation == 1 and total_updates == 6000
    assert len(base_loader.dataset) == protocol['expected_training_images']
    assert Counter(m['task'] for m in base_loader.dataset.coco.imgs.values()) == protocol['expected_task_counts']
    assert validation_interval == 0

    runner = {"condition": condition, "seed": seed, "successful_updates": 0, "microbatches_consumed": 0, "amp_skipped_updates": 0, "total_exposures": 0, "exposures_by_domain": {}, "exposures_by_task": {}, "checkpoint_selection": "none"}
    metrics_path = output / "metrics.jsonl"
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["identities"] != identities:
            raise RuntimeError("resume identity mismatch")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        runner = state["runner"]
        restore_rng(state["rng"])
        assert_state_equal(model.state_dict(), state["model"])
        assert_state_equal(ema.state_dict(), state["ema"])
        assert_state_equal(optimizer.state_dict(), state["optimizer"])
        assert_state_equal(scaler.state_dict(), state["scaler"])
        assert_state_equal(capture_rng(), state["rng"])
        expected_schedule = {"warmup_updates": warmup, "lr_drop_update": drop, "lr_drop_factor": gamma, "base_group_lrs": base_group_lrs, "successful_update_semantics": True}
        assert_state_equal(expected_schedule, state["custom_update_schedule"])
        atomic_json(output / "RESUME_RESTORE_AUDIT.json", {"status": "pass", "source_checkpoint_sha256": sha256(args.resume), "restored_successful_updates": runner["successful_updates"], "restored_microbatch_position": runner["microbatches_consumed"], "exact_restoration": ["model", "EMA", "optimizer", "scaler", "RNG", "scheduler"], "next_batch_indices": StatelessRepeatBatchSampler(len(base_loader.dataset), physical_batch, seed).batch(runner["microbatches_consumed"])})
        del state
        trim_metrics_for_resume(metrics_path, runner)
    elif metrics_path.exists():
        raise RuntimeError(f"nonempty output prevents accidental overwrite: {metrics_path}")

    if args.probe:
        sampler = ProbeBatchSampler(base_loader.dataset, physical_batch)
    else:
        sampler = StatelessRepeatBatchSampler(len(base_loader.dataset), physical_batch, seed, int(runner["microbatches_consumed"]))
    train_loader = make_loader(base_loader, sampler, seed)
    iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    update_times = []

    def save(name: str) -> None:
        atomic_checkpoint(output / name, checkpoint_payload(model, ema, optimizer, scaler, runner, identities, {"warmup_updates": warmup, "lr_drop_update": drop, "lr_drop_factor": gamma, "base_group_lrs": base_group_lrs, "successful_update_semantics": True}))

    if not args.probe and not args.resume:
        save("last.pth")

    if args.smoke_limit:
        total_updates = args.smoke_limit
    probe_target = 2
    while (int(runner["successful_updates"]) < total_updates) if not args.probe else (int(runner["successful_updates"]) < probe_target):
        tick = time.perf_counter()
        samples, targets = next(iterator)
        runner["microbatches_consumed"] = int(runner["microbatches_consumed"]) + 1
        runner["total_exposures"] = int(runner["total_exposures"]) + int(samples.shape[0])
        batch_domains: Counter[str] = Counter()
        batch_tasks: Counter[str] = Counter()
        for target in targets:
            idx = int(target["idx"].reshape(-1)[0])
            meta = base_loader.dataset.coco.imgs[base_loader.dataset.ids[idx]]
            batch_domains[meta["domain"]] += 1
            batch_tasks[meta["task"]] += 1
        for key, value in batch_domains.items():
            runner["exposures_by_domain"][key] = int(runner["exposures_by_domain"].get(key, 0)) + value
        for key, value in batch_tasks.items():
            runner["exposures_by_task"][key] = int(runner["exposures_by_task"].get(key, 0)) + value

        next_update = int(runner["successful_updates"]) + 1
        factor = schedule_factor(next_update, warmup, drop, gamma)
        for group, base_lr in zip(optimizer.param_groups, base_group_lrs):
            group["lr"] = base_lr * factor
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = criterion(outputs, targets, epoch=0, step=0, global_step=int(runner["successful_updates"]))
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss_dict}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_max_norm)).detach().cpu())
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        skipped = new_scale < old_scale
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - tick
        update_times.append(seconds)
        components = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
        if skipped:
            runner["amp_skipped_updates"] = int(runner["amp_skipped_updates"]) + 1
            kind = "amp_skipped"
        else:
            runner["successful_updates"] = next_update
            ema.update(model)
            kind = "train"
        record = {"kind": kind, "condition": condition, "optimizer_update": int(runner["successful_updates"]), "microbatch_attempt": int(runner["microbatches_consumed"]), "samples": int(samples.shape[0]), "batch_domains": dict(batch_domains), "batch_tasks": dict(batch_tasks), "cumulative_exposures": int(runner["total_exposures"]), "cumulative_exposures_by_domain": dict(runner["exposures_by_domain"]), "cumulative_exposures_by_task": dict(runner["exposures_by_task"]), "loss": float(loss.detach().cpu()), "loss_components": components, "grad_norm_before_clip": grad_norm, "lr": [float(g["lr"]) for g in optimizer.param_groups], "amp_scale_before": old_scale, "amp_scale_after": new_scale, "update_seconds": seconds, "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
        if not args.probe:
            append_jsonl(metrics_path, record)
        if args.probe or int(runner["successful_updates"]) % int(cfg.print_freq) == 0 or int(runner["successful_updates"]) == 1:
            print("TRAIN=" + json.dumps(record, sort_keys=True), flush=True)

        completed = int(runner["successful_updates"])
        if not args.probe and not skipped and (completed % 1000 == 0 or completed == total_updates):
            save("last.pth")
            print(f"CHECKPOINT seed={seed} successful_updates={completed}", flush=True)

    elapsed = time.perf_counter() - start
    summary = {"status": "probe_pass" if args.probe else "pass", "condition": condition, "successful_updates": int(runner["successful_updates"]), "amp_skipped_updates": int(runner["amp_skipped_updates"]), "physical_batch": physical_batch, "accumulation": accumulation, "effective_batch": effective_batch, "total_exposures": int(runner["total_exposures"]), "exposures_by_domain": runner["exposures_by_domain"], "exposures_by_task": runner["exposures_by_task"], "elapsed_seconds": elapsed, "mean_update_seconds": float(np.mean(update_times)), "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "gpu": torch.cuda.get_device_name(device), "numerical_stability": "finite losses; no AMP skips" if int(runner["amp_skipped_updates"]) == 0 else "finite losses with AMP skipped updates", "identities": identities}
    if args.probe:
        atomic_json(output / "probe_result.json", summary)
    else:
        save("last.pth")
        atomic_json(output / "run_summary.json", summary)
        atomic_json(output / ("SMOKE_COMPLETE.json" if args.smoke_limit else "COMPLETE.json"), {"status": "complete", "condition": condition, "updates": total_updates})
        validations = [json.loads(x) for x in metrics_path.read_text().splitlines() if json.loads(x).get("kind") == "validation"]
        csv_path = output / "validation_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["optimizer_update", *METRIC_NAMES, "weights", "validation_seconds"])
            writer.writeheader()
            for v in validations:
                writer.writerow({"optimizer_update": v["optimizer_update"], **v["metrics"], "weights": v["weights"], "validation_seconds": v["validation_seconds"]})
    print("SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
