from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Sequence

Hypothesis = Hashable
Action = Hashable
Predictor = Callable[[Hypothesis, Action], Hashable]


def canonical_partition(sizes: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(x) for x in sizes)
    if not values:
        raise ValueError("partition must contain at least one block")
    if any(x <= 0 for x in values):
        raise ValueError("partition blocks must be positive")
    return tuple(sorted(values, reverse=True))


def partition_distance(left: Iterable[int], right: Iterable[int]) -> int:
    a = canonical_partition(left)
    b = canonical_partition(right)
    width = max(len(a), len(b))
    aa = a + (0,) * (width - len(a))
    bb = b + (0,) * (width - len(b))
    return sum(abs(x - y) for x, y in zip(aa, bb))


def structural_partition(
    hypotheses: Sequence[Hypothesis],
    action: Action,
    predictor: Predictor,
) -> tuple[frozenset[Hypothesis], ...]:
    groups: dict[Hashable, set[Hypothesis]] = {}
    for h in hypotheses:
        outcome = predictor(h, action)
        groups.setdefault(outcome, set()).add(h)
    blocks = [frozenset(group) for group in groups.values() if group]
    return tuple(sorted(blocks, key=lambda b: (-len(b), tuple(sorted(map(repr, b))))))


def partition_hypotheses(
    hypotheses: Sequence[Hypothesis],
    action: Action,
    predictor: Predictor,
) -> tuple[int, ...]:
    return canonical_partition(
        len(block) for block in structural_partition(hypotheses, action, predictor)
    )


@dataclass(frozen=True)
class BalanceScore:
    action: Action
    partition: tuple[int, ...]
    distance: int


@dataclass(frozen=True)
class BalanceDecision:
    target_partition: tuple[int, ...]
    selected_action: Action
    selected_partition: tuple[int, ...]
    distance: int
    ranking: tuple[BalanceScore, ...]
    exact_matches: tuple[Action, ...]

    @property
    def exact(self) -> bool:
        return self.distance == 0


def select_balance(
    hypotheses: Sequence[Hypothesis],
    actions: Sequence[Action],
    target_partition: Iterable[int],
    predictor: Predictor,
) -> BalanceDecision:
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    if not actions:
        raise ValueError("actions must not be empty")

    target = canonical_partition(target_partition)
    if sum(target) != len(hypotheses):
        raise ValueError("target partition must sum to the number of hypotheses")

    scores: list[BalanceScore] = []
    for action in actions:
        part = partition_hypotheses(hypotheses, action, predictor)
        scores.append(
            BalanceScore(
                action=action,
                partition=part,
                distance=partition_distance(part, target),
            )
        )

    ranking = tuple(sorted(scores, key=lambda score: (score.distance, repr(score.action))))
    winner = ranking[0]
    exact_matches = tuple(score.action for score in ranking if score.distance == 0)
    return BalanceDecision(
        target_partition=target,
        selected_action=winner.action,
        selected_partition=winner.partition,
        distance=winner.distance,
        ranking=ranking,
        exact_matches=exact_matches,
    )
