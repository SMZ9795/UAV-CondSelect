from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from source_balanced_sampler import (
    EpochRandomSampler,
    FixedSecondViewSampler,
    SourceBalancedSampler,
)
from uav_common import (
    DEFAULT_DATA_ROOT,
    MODEL_COMMIT,
    add_upstream_to_path,
    coco_metrics,
    config_sha256,
    load_model_weights,
    seed_everything,
    verify_checkpoint,
    verify_upstream_commit,
    write_json,
)


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_train_loader(base_loader, sampler, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": base_loader.dataset,
        "batch_size": base_loader.batch_size,
        "sampler": sampler,
        "num_workers": base_loader.num_workers,
        "collate_fn": base_loader.collate_fn,
        "pin_memory": base_loader.pin_memory,
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": partial(seed_worker, base_seed=seed),
    }
    if base_loader.num_workers > 0:
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_state(
    model,
    ema,
    optimizer,
    lr_scheduler,
    warmup,
    scaler,
    runner_state: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "runner": dict(runner_state),
        "rng_state": capture_rng_state(),
        "model_commit": MODEL_COMMIT,
        "config_sha256": config_sha256(config_path),
        "optimizer": optimizer.state_dict(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if warmup is not None:
        state["lr_warmup_scheduler"] = warmup.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    return state


def save_checkpoint(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_state(**kwargs), temporary)
    temporary.replace(path)


def load_resume(
    path: Path,
    model,
    ema,
    optimizer,
    lr_scheduler,
    warmup,
    scaler,
    budget_total_updates: int,
    config_path: Path,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("model_commit") != MODEL_COMMIT:
        raise RuntimeError("Resume checkpoint model commit mismatch")
    expected_config_sha256 = config_sha256(config_path)
    if state.get("config_sha256") != expected_config_sha256:
        raise RuntimeError(
            "Resume checkpoint config hash mismatch: "
            f"expected {expected_config_sha256}, found {state.get('config_sha256')}"
        )
    model.load_state_dict(state["model"], strict=True)
    if ema is not None:
        ema.load_state_dict(state["ema"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    if warmup is not None and "lr_warmup_scheduler" in state:
        warmup.load_state_dict(state["lr_warmup_scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    runner = state["runner"]
    if int(runner["budget_total_updates"]) != budget_total_updates:
        raise RuntimeError(
            "Cannot change total optimizer-update budget on resume because it defines the update-based augmentation schedule"
        )
    restore_rng_state(state["rng_state"])
    return runner


def prepare_curve_for_resume(curve_path: Path, resume_update: int) -> list[int]:
    if not curve_path.is_file():
        raise RuntimeError(
            f"Resume checkpoint is at update {resume_update}, but the learning curve is missing"
        )
    records = [
        json.loads(line)
        for line in curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_records = [record for record in records if record.get("kind") == "train"]
    retained_train = [
        record for record in train_records if int(record["optimizer_update"]) <= resume_update
    ]
    expected_updates = list(range(1, resume_update + 1))
    actual_updates = [int(record["optimizer_update"]) for record in retained_train]
    if actual_updates != expected_updates:
        raise RuntimeError(
            "Learning-curve prefix does not match the resume checkpoint: "
            f"expected updates 1..{resume_update}, found {actual_updates[:5]}..."
        )
    retained = [
        record
        for record in records
        if int(record.get("optimizer_update", 0)) <= resume_update
    ]
    discarded = [
        record
        for record in records
        if int(record.get("optimizer_update", 0)) > resume_update
    ]
    if discarded:
        tail_path = curve_path.with_name(
            f"learning_curve.interrupted_tail_after_update_{resume_update}.jsonl"
        )
        with tail_path.open("a", encoding="utf-8") as handle:
            for record in discarded:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    with curve_path.open("w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return [int(record["samples"]) for record in retained_train]


def evaluate_val_clean(cfg, model, ema, device) -> tuple[dict[str, float], list[float]]:
    from src.solver.det_engine import evaluate

    val_loader = cfg.val_dataloader
    annotation = Path(val_loader.dataset.ann_file).name
    if annotation != "instances_val_clean.json":
        raise RuntimeError(
            f"Checkpoint selection is restricted to val_clean, found {annotation}"
        )
    val_loader.dataset.set_epoch(10**9)
    val_loader.collate_fn.set_epoch(10**9)
    module = ema.module if ema is not None else model
    stats, _ = evaluate(
        module,
        cfg.criterion,
        cfg.postprocessor,
        val_loader,
        cfg.evaluator,
        device,
    )
    values = stats["coco_eval_bbox"]
    return coco_metrics(values), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-final-validation", action="store_true")
    args = parser.parse_args()

    verify_upstream_commit()
    add_upstream_to_path()
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config.resolve()))
    protocol = cfg.yaml_cfg.get("uav_protocol", {})
    experiment = protocol.get("experiment")
    if experiment not in {"E0", "E1", "STAGE2A"}:
        raise RuntimeError(f"Unknown uav_protocol experiment: {experiment}")
    seed = int(protocol["sampler_seed"])
    if seed not in {0, 1, 2, 3}:
        raise RuntimeError("Formal/smoke seeds are restricted to 0, 1, 2, or 3")
    budget_total_updates = int(protocol["total_optimizer_updates"])
    stop_after = (
        int(args.stop_after_updates)
        if args.stop_after_updates is not None
        else budget_total_updates
    )
    if not (0 < stop_after <= budget_total_updates):
        raise ValueError("stop-after-updates must be in [1, total_optimizer_updates]")
    accumulation = int(protocol["gradient_accumulation_steps"])
    validation_interval = int(protocol["validation_interval_updates"])
    output_dir = (args.output_dir or Path(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = cfg.model
    checkpoint_path = Path(cfg.tuning)
    verify_checkpoint(checkpoint_path)
    if args.resume is None:
        preload_report = load_model_weights(model, checkpoint_path)
    else:
        preload_report = {
            "skipped_due_to_resume": True,
            "resume": str(args.resume.resolve()),
        }
    write_json(output_dir / "pretrained_load_report.json", preload_report)
    print("PRETRAINED_LOAD_REPORT=" + json.dumps(preload_report, sort_keys=True))

    model = model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    optimizer = cfg.optimizer
    lr_scheduler = cfg.lr_scheduler
    warmup = cfg.lr_warmup_scheduler
    ema = cfg.ema
    if ema is not None:
        ema.to(device)
    scaler = cfg.scaler if cfg.yaml_cfg.get("use_amp", False) else None
    use_amp = scaler is not None and device.type == "cuda"

    base_loader = cfg.train_dataloader
    full_pseudo_epoch_samples = (
        204 if experiment in {"E1", "STAGE2A"} else len(base_loader.dataset)
    )
    physical_batch_size = int(base_loader.batch_size)
    require_full_train_batches = bool(
        protocol.get("require_full_train_batches", False)
    )
    configured_drop_last = bool(
        cfg.yaml_cfg["train_dataloader"].get("drop_last", False)
    )
    if require_full_train_batches:
        if configured_drop_last:
            raise RuntimeError(
                "Fair protocol requires train_dataloader.drop_last=false"
            )
        if full_pseudo_epoch_samples % physical_batch_size != 0:
            raise RuntimeError(
                "Fair protocol requires the physical batch size to divide the "
                f"full pseudo-epoch exactly: {full_pseudo_epoch_samples} % "
                f"{physical_batch_size} != 0"
            )
    if experiment == "STAGE2A":
        sampler = FixedSecondViewSampler(
            base_loader.dataset,
            args.dataset_root,
            seed,
            protocol["second_view"],
        )
    elif experiment == "E1":
        sampler = SourceBalancedSampler(base_loader.dataset, args.dataset_root, seed)
    else:
        sampler = EpochRandomSampler(base_loader.dataset, seed)
    train_loader = build_train_loader(base_loader, sampler, seed)

    runner_state: dict[str, Any] = {
        "experiment": experiment,
        "seed": seed,
        "optimizer_updates": 0,
        "pseudo_epoch": 0,
        "samples_consumed_in_epoch": 0,
        "budget_total_updates": budget_total_updates,
        "best_ap": -1.0,
        "best_update": -1,
    }
    runner_state.update(successful_updates=0, amp_skips=0)
    resumed_from = None
    if args.resume is not None:
        resumed_from = str(args.resume.resolve())
        runner_state = load_resume(
            args.resume.resolve(),
            model,
            ema,
            optimizer,
            lr_scheduler,
            warmup,
            scaler,
            budget_total_updates,
            args.config.resolve(),
        )
    if int(runner_state["optimizer_updates"]) >= stop_after:
        raise RuntimeError("Resume checkpoint is already at or beyond requested stop update")

    optimizer.zero_grad(set_to_none=True)
    micro_in_accumulation = 0
    update_times: list[float] = []
    update_sample_counts: list[int] = []
    total_training_images = 0
    microbatches_seen = 0
    partial_microbatches_observed = 0
    last_metrics: dict[str, float] | None = None
    start_update = int(runner_state["optimizer_updates"])
    start_time = time.perf_counter()
    curve_path = output_dir / "learning_curve.jsonl"
    if args.resume is None:
        curve_path.unlink(missing_ok=True)
    else:
        update_sample_counts = prepare_curve_for_resume(curve_path, start_update)
        total_training_images = sum(update_sample_counts)
        microbatches_seen = len(update_sample_counts) * accumulation
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def append_curve(record: dict[str, Any]) -> None:
        with curve_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def save(name: str) -> None:
        save_checkpoint(
            output_dir / name,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            warmup=warmup,
            scaler=scaler,
            runner_state=runner_state,
            config_path=args.config.resolve(),
        )

    while int(runner_state["optimizer_updates"]) < stop_after:
        pseudo_epoch = int(runner_state["pseudo_epoch"])
        offset = int(runner_state["samples_consumed_in_epoch"])
        sampler.set_epoch(pseudo_epoch)
        sampler.set_start_offset(offset)
        augmentation_epoch = min(
            int(protocol["augmentation_reference_epochs"]) - 1,
            int(
                int(runner_state["optimizer_updates"])
                * int(protocol["augmentation_reference_epochs"])
                / budget_total_updates
            ),
        )
        train_loader.dataset.set_epoch(augmentation_epoch)
        train_loader.collate_fn.set_epoch(augmentation_epoch)
        exhausted = True

        for samples, targets in train_loader:
            exhausted = False
            if micro_in_accumulation == 0:
                update_tick = time.perf_counter()
                update_samples = 0
                accumulated_loss = 0.0
                accumulated_components: dict[str, float] = {}
            microbatch_samples = int(samples.shape[0])
            microbatches_seen += 1
            if microbatch_samples != physical_batch_size:
                partial_microbatches_observed += 1
                if require_full_train_batches:
                    raise RuntimeError(
                        "Fair protocol encountered a partial training batch: "
                        f"expected {physical_batch_size}, observed {microbatch_samples} "
                        f"in {experiment} pseudo-epoch {pseudo_epoch}"
                    )
            samples = samples.to(device, non_blocking=True)
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            current_update = int(runner_state["optimizer_updates"])
            metas = {
                "epoch": augmentation_epoch,
                "step": micro_in_accumulation,
                "global_step": current_update,
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
                cache_enabled=True,
            ):
                outputs = model(samples, targets=targets)
            with torch.autocast(device_type=device.type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
                unscaled_loss = sum(loss_dict.values())
                loss = unscaled_loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {loss_dict}")
            accumulated_loss += float(unscaled_loss.detach().cpu()) / accumulation
            for name, value in loss_dict.items():
                accumulated_components[name] = accumulated_components.get(name, 0.0) + (
                    float(value.detach().cpu()) / accumulation
                )
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_in_accumulation += 1
            runner_state["samples_consumed_in_epoch"] = (
                int(runner_state["samples_consumed_in_epoch"]) + int(samples.shape[0])
            )
            update_samples += int(samples.shape[0])

            if micro_in_accumulation < accumulation:
                continue
            expected_update_samples = physical_batch_size * accumulation
            if require_full_train_batches and update_samples != expected_update_samples:
                raise RuntimeError(
                    "Fair protocol optimizer-update sample count mismatch: "
                    f"expected {expected_update_samples}, observed {update_samples}"
                )
            if scaler is not None:
                scaler.unscale_(optimizer)
            if cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            if scaler is not None:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < scale_before
                runner_state["amp_skips"] += int(skipped)
                runner_state["successful_updates"] += int(not skipped)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accumulation = 0
            runner_state["optimizer_updates"] = current_update + 1
            if ema is not None:
                ema.update(model)
            if warmup is not None and not warmup.finished():
                warmup.step()
            elif lr_scheduler is not None:
                lr_scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_tick
            update_times.append(update_seconds)
            update_sample_counts.append(update_samples)
            total_training_images += update_samples

            completed = int(runner_state["optimizer_updates"])
            training_record = {
                "kind": "train",
                "attempted_updates": runner_state["optimizer_updates"],
                "successful_updates": runner_state["successful_updates"],
                "amp_skips": runner_state["amp_skips"],
                "experiment": experiment,
                "optimizer_update": completed,
                "pseudo_epoch": pseudo_epoch,
                "augmentation_epoch": augmentation_epoch,
                "loss": accumulated_loss,
                "loss_components": accumulated_components,
                "lr": [group["lr"] for group in optimizer.param_groups],
                "samples": update_samples,
                "update_seconds": update_seconds,
                "images_per_second": update_samples / update_seconds,
            }
            append_curve(training_record)
            if completed == 1 or completed % max(1, int(cfg.print_freq)) == 0:
                print(json.dumps(training_record, sort_keys=True))

            should_validate = completed % validation_interval == 0
            is_stop = completed >= stop_after
            if should_validate or (is_stop and not args.no_final_validation):
                validation_tick = time.perf_counter()
                last_metrics, _ = evaluate_val_clean(cfg, model, ema, device)
                validation_seconds = time.perf_counter() - validation_tick
                append_curve(
                    {
                        "kind": "validation",
                        "subset": "val_clean",
                        "experiment": experiment,
                        "optimizer_update": completed,
                        "metrics": last_metrics,
                        "validation_seconds": validation_seconds,
                    }
                )
                print("VAL_CLEAN=" + json.dumps(last_metrics, sort_keys=True))
                if last_metrics["AP"] > float(runner_state["best_ap"]):
                    runner_state["best_ap"] = last_metrics["AP"]
                    runner_state["best_update"] = completed
                    save("best.pth")
                save("last.pth")
            if is_stop:
                break

        samples_consumed = int(runner_state["samples_consumed_in_epoch"])
        if samples_consumed > full_pseudo_epoch_samples:
            raise RuntimeError(
                "Sampler consumed beyond the pseudo-epoch boundary: "
                f"{samples_consumed} > {full_pseudo_epoch_samples}"
            )
        if samples_consumed == full_pseudo_epoch_samples:
            runner_state["pseudo_epoch"] = int(runner_state["pseudo_epoch"]) + 1
            runner_state["samples_consumed_in_epoch"] = 0
        elif exhausted:
            raise RuntimeError("Dataloader exhausted without consuming samples")

    if not (output_dir / "last.pth").is_file():
        save("last.pth")
    elapsed = time.perf_counter() - start_time
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    summary = {
        "status": "pass",
        "experiment": experiment,
        "seed": seed,
        "config": str(args.config.resolve()),
        "config_sha256": config_sha256(args.config),
        "model_commit": MODEL_COMMIT,
        "physical_batch_size": base_loader.batch_size,
        "configured_drop_last": configured_drop_last,
        "require_full_train_batches": require_full_train_batches,
        "full_pseudo_epoch_samples": full_pseudo_epoch_samples,
        "full_batches_per_pseudoepoch": (
            full_pseudo_epoch_samples // physical_batch_size
        ),
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch_size_single_gpu": base_loader.batch_size * accumulation,
        "expected_images_per_optimizer_update": physical_batch_size * accumulation,
        "start_optimizer_update": start_update,
        "completed_optimizer_updates": int(runner_state["optimizer_updates"]),
        "budget_total_optimizer_updates": budget_total_updates,
        "resumed_from": resumed_from,
        "checkpoint_last": str((output_dir / "last.pth").resolve()),
        "checkpoint_best": (
            str((output_dir / "best.pth").resolve())
            if (output_dir / "best.pth").is_file()
            else None
        ),
        "best_val_clean_ap": float(runner_state["best_ap"]),
        "best_update": int(runner_state["best_update"]),
        "last_val_clean_metrics": last_metrics,
        "elapsed_seconds": elapsed,
        "learning_curve": str(curve_path.resolve()),
        "training_images": total_training_images,
        "expected_training_images_for_completed_updates": (
            (int(runner_state["optimizer_updates"]) - start_update)
            * physical_batch_size
            * accumulation
        ),
        "minimum_images_per_optimizer_update": (
            min(update_sample_counts) if update_sample_counts else None
        ),
        "maximum_images_per_optimizer_update": (
            max(update_sample_counts) if update_sample_counts else None
        ),
        "microbatches_seen": microbatches_seen,
        "partial_microbatches_observed": partial_microbatches_observed,
        "completed_full_pseudoepochs": int(runner_state["pseudo_epoch"]),
        "final_pseudoepoch_samples_consumed": int(
            runner_state["samples_consumed_in_epoch"]
        ),
        "dropped_samples_per_complete_pseudoepoch": 0,
        "training_update_seconds": float(sum(update_times)),
        "mean_training_images_per_second": (
            float(total_training_images / sum(update_times)) if update_times else None
        ),
        "mean_optimizer_update_seconds": (
            float(np.mean(update_times)) if update_times else None
        ),
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "amp_enabled": use_amp,
        "amp_dtype": "float16" if use_amp else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(output_dir / "run_summary.json", summary)
    print("RUN_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
