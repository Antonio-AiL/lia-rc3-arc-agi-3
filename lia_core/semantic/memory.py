from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable, Mapping, Sequence


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"


def _normalize_blocks(
    partition: Iterable[Iterable[Hashable]],
) -> tuple[frozenset[Hashable], ...]:
    blocks = [frozenset(block) for block in partition]
    return tuple(
        sorted(
            blocks,
            key=lambda block: (-len(block), tuple(sorted(map(repr, block)))),
        )
    )


def _partition_integrity(
    hypothesis_ids: Sequence[Hashable],
    blocks: tuple[frozenset[Hashable], ...],
) -> bool:
    expected = set(hypothesis_ids)
    observed: set[Hashable] = set()
    for block in blocks:
        if not block or observed.intersection(block):
            return False
        observed.update(block)
    return observed == expected


def evaluate_sufficiency(
    hypothesis_ids: Sequence[Hashable],
    partition: Iterable[Iterable[Hashable]],
    task: Mapping[Hashable, Hashable] | None,
    *,
    conflict: bool = False,
) -> SufficiencyStatus:
    blocks = _normalize_blocks(partition)
    if conflict or not _partition_integrity(hypothesis_ids, blocks):
        return SufficiencyStatus.CONFLICT
    if task is None:
        return SufficiencyStatus.UNRESOLVED
    if set(task) != set(hypothesis_ids):
        return SufficiencyStatus.CONFLICT

    for block in blocks:
        if len({task[h] for h in block}) > 1:
            return SufficiencyStatus.UNRESOLVED
    return SufficiencyStatus.SUFFICIENT


@dataclass(frozen=True)
class SemanticMemoryState:
    hypothesis_ids: tuple[Hashable, ...]
    partitions: tuple[tuple[str, tuple[tuple[Hashable, ...], ...]], ...]
    relevant_distinctions: tuple[tuple[Hashable, Hashable], ...]
    preserved_relations: tuple[tuple[str, str, str], ...]
    sufficiency_status: SufficiencyStatus
    evidence: tuple[str, ...]
    version_id: str
    parent_version: str | None = None


def build_semantic_state(
    hypothesis_ids: Sequence[Hashable],
    *,
    named_partition: tuple[str, Iterable[Iterable[Hashable]]],
    task: Mapping[Hashable, Hashable] | None,
    relations: Iterable[tuple[str, str, str]] = (),
    evidence: Iterable[str] = (),
    parent_version: str | None = None,
    conflict: bool = False,
) -> SemanticMemoryState:
    name, raw_partition = named_partition
    blocks = _normalize_blocks(raw_partition)
    relation_items = tuple(sorted(tuple(map(str, rel)) for rel in relations))
    evidence_items = tuple(map(str, evidence))
    status = evaluate_sufficiency(hypothesis_ids, blocks, task, conflict=conflict)

    relevant: list[tuple[Hashable, Hashable]] = []
    if task is not None and status is not SufficiencyStatus.CONFLICT:
        for block in blocks:
            ordered = sorted(block, key=repr)
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    if task[left] != task[right]:
                        relevant.append((left, right))

    serial_blocks = tuple(tuple(sorted(block, key=repr)) for block in blocks)
    payload = {
        "hypothesis_ids": list(map(repr, hypothesis_ids)),
        "partition_name": str(name),
        "partition": [[repr(x) for x in block] for block in serial_blocks],
        "task": None if task is None else sorted((repr(k), repr(v)) for k, v in task.items()),
        "relations": relation_items,
        "evidence": evidence_items,
        "parent_version": parent_version,
        "status": status.value,
    }
    version_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    return SemanticMemoryState(
        hypothesis_ids=tuple(hypothesis_ids),
        partitions=((str(name), serial_blocks),),
        relevant_distinctions=tuple(relevant),
        preserved_relations=relation_items,
        sufficiency_status=status,
        evidence=evidence_items,
        version_id=version_id,
        parent_version=parent_version,
    )
