from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterator

from torch.utils.data import Sampler

from uav_common import stable_int


VARIANTS = ("rain", "snow", "fog", "dust", "dim_low_light")
ALL_VERSIONS = ("real",) + VARIANTS
FILENAME_PATTERN = re.compile(
    r"^(?P<source>.+)_(?P<version>real|rain|snow|fog|dust|dim_low_light)$"
)


def parse_filename(file_name: str) -> tuple[str, str]:
    match = FILENAME_PATTERN.fullmatch(Path(file_name).stem)
    if match is None:
        raise ValueError(f"Non-canonical UAV dataset filename: {file_name}")
    return match.group("source"), match.group("version")


def read_source_ids(path: Path) -> set[str]:
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    return values


def dataset_index(dataset) -> dict[str, dict[str, int]]:
    mapping: dict[str, dict[str, int]] = {}
    for index, image_id in enumerate(dataset.ids):
        meta = dataset.coco.imgs[image_id]
        file_name = meta["file_name"]
        source, version = (meta["source_id"], meta["version"]) if "source_id" in meta and "version" in meta else parse_filename(file_name)
        versions = mapping.setdefault(source, {})
        if version in versions:
            raise ValueError(f"Duplicate {version} image for source {source}")
        versions[version] = index
    return mapping


def assert_train_only(mapping: dict[str, dict[str, int]], dataset_root: Path) -> None:
    train = read_source_ids(dataset_root / "splits" / "train_sources.txt")
    val = read_source_ids(dataset_root / "splits" / "val_sources.txt")
    test = read_source_ids(dataset_root / "splits" / "test_sources.txt")
    observed = set(mapping)
    leaked = observed & (val | test)
    if leaked:
        raise ValueError(f"Validation/test leakage in training dataset: {sorted(leaked)}")
    if observed != train:
        raise ValueError(
            "Training source set mismatch: "
            f"missing={sorted(train - observed)}, unexpected={sorted(observed - train)}"
        )


class EpochRandomSampler(Sampler[int]):
    """One deterministic permutation of every dataset index per pseudo-epoch."""

    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.start_offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        if offset < 0 or offset > len(self.dataset):
            raise ValueError(f"Invalid E0 sampler offset: {offset}")
        self.start_offset = int(offset)

    def indices_for_epoch(self, epoch: int) -> list[int]:
        indices = list(range(len(self.dataset)))
        random.Random(stable_int("e0-order", self.seed, epoch)).shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices_for_epoch(self.epoch)[self.start_offset :])

    def __len__(self) -> int:
        return len(self.dataset) - self.start_offset


class SourceBalancedSampler(Sampler[int]):
    """Deterministic 102-real + 102-synthetic source-balanced sampler."""

    def __init__(self, dataset, dataset_root: Path, seed: int):
        self.dataset = dataset
        self.dataset_root = dataset_root.resolve()
        self.seed = int(seed)
        self.epoch = 0
        self.start_offset = 0
        self.mapping = dataset_index(dataset)
        assert_train_only(self.mapping, self.dataset_root)
        if len(self.mapping) != 102:
            raise ValueError(f"Expected 102 training sources, found {len(self.mapping)}")
        for source, versions in self.mapping.items():
            missing = {"real", self.second_view} - set(versions)
            extra = set()
            if missing or extra:
                raise ValueError(
                    f"Source {source} versions invalid: missing={missing}, extra={extra}"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        full_length = 2 * len(self.mapping)
        if offset < 0 or offset > full_length:
            raise ValueError(f"Invalid E1 sampler offset: {offset}")
        self.start_offset = int(offset)

    def weather_order(self, source: str) -> tuple[str, ...]:
        order = list(VARIANTS)
        random.Random(stable_int("weather-cycle", self.seed, source)).shuffle(order)
        return tuple(order)

    def selected_weather(self, source: str, epoch: int) -> str:
        return self.weather_order(source)[epoch % len(VARIANTS)]

    def records_for_epoch(self, epoch: int) -> list[tuple[int, str, str]]:
        records: list[tuple[int, str, str]] = []
        for source in sorted(self.mapping):
            records.append((self.mapping[source]["real"], source, "real"))
            weather = self.selected_weather(source, epoch)
            records.append((self.mapping[source][weather], source, weather))
        random.Random(stable_int("e1-order", self.seed, epoch)).shuffle(records)
        return records

    def indices_for_epoch(self, epoch: int) -> list[int]:
        return [index for index, _, _ in self.records_for_epoch(epoch)]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices_for_epoch(self.epoch)[self.start_offset :])

    def __len__(self) -> int:
        return 2 * len(self.mapping) - self.start_offset


class FixedSecondViewSampler(Sampler[int]):
    """Deterministic Stage 2A stream: clean plus one frozen second view/source."""

    def __init__(self, dataset, dataset_root: Path, seed: int, second_view: str):
        self.dataset = dataset
        self.dataset_root = dataset_root.resolve()
        self.seed = int(seed)
        self.second_view = str(second_view)
        if self.second_view not in ALL_VERSIONS:
            raise ValueError(
                f"Stage 2A second_view must be one of {ALL_VERSIONS}, "
                f"found {self.second_view!r}"
            )
        self.epoch = 0
        self.start_offset = 0
        self.mapping = dataset_index(dataset)
        assert_train_only(self.mapping, self.dataset_root)
        if len(self.mapping) != 102:
            raise ValueError(f"Expected 102 training sources, found {len(self.mapping)}")
        for source, versions in self.mapping.items():
            missing = {"real", self.second_view} - set(versions)
            extra = set()
            if missing or extra:
                raise ValueError(
                    f"Source {source} versions invalid: missing={missing}, extra={extra}"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        full_length = 2 * len(self.mapping)
        if offset < 0 or offset > full_length:
            raise ValueError(f"Invalid Stage 2A sampler offset: {offset}")
        self.start_offset = int(offset)

    def records_for_epoch(self, epoch: int) -> list[tuple[int, str, str]]:
        records: list[tuple[int, str, str]] = []
        for source in sorted(self.mapping):
            records.append((self.mapping[source]["real"], source, "real"))
            records.append(
                (self.mapping[source][self.second_view], source, self.second_view)
            )
        # The ordering key deliberately excludes second_view so all five models
        # receive the same source/occurrence schedule for a given seed and epoch.
        random.Random(stable_int("stage2a-order", self.seed, epoch)).shuffle(records)
        return records

    def indices_for_epoch(self, epoch: int) -> list[int]:
        return [index for index, _, _ in self.records_for_epoch(epoch)]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices_for_epoch(self.epoch)[self.start_offset :])

    def __len__(self) -> int:
        return 2 * len(self.mapping) - self.start_offset
