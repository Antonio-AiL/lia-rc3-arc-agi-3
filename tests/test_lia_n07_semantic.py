from __future__ import annotations

import unittest

import numpy as np

from agent.my_agent import AOAArc3Core, Obs
from agent.my_agent_semantic import SemanticAOAArc3Core
from lia_core.semantic import (
    SufficiencyStatus,
    build_semantic_state,
    select_balance,
    structural_partition,
)


HYPOTHESES = tuple(f"h{i:02d}" for i in range(22))
ACTIONS = ("A03", "A07", "A09")


def fixture_predict(h: str, action: str) -> str:
    i = int(h[1:])
    if action == "A03":
        return "left" if i < 11 else "right"
    if action == "A07":
        return "even" if i % 2 == 0 else "odd"
    if action == "A09":
        if i < 12:
            return "p0"
        if i < 17:
            return "p1"
        return "p2"
    raise KeyError(action)


def reference_partition(action: str) -> tuple[int, ...]:
    # Independent oracle: deliberately does not call production partition helpers.
    counts: dict[str, int] = {}
    for h in HYPOTHESES:
        outcome = fixture_predict(h, action)
        counts[outcome] = counts.get(outcome, 0) + 1
    return tuple(sorted(counts.values(), reverse=True))


class ReverseBalanceTests(unittest.TestCase):
    def test_t03_preserves_all_exact_matches_and_canonical_winner(self) -> None:
        self.assertEqual(reference_partition("A03"), (11, 11))
        self.assertEqual(reference_partition("A07"), (11, 11))

        decision = select_balance(HYPOTHESES, ACTIONS, (11, 11), fixture_predict)

        self.assertTrue(decision.exact)
        self.assertEqual(decision.distance, 0)
        self.assertEqual(decision.selected_action, "A03")
        self.assertEqual(decision.exact_matches, ("A03", "A07"))
        self.assertEqual(decision.selected_partition, (11, 11))
        for score in decision.ranking:
            self.assertEqual(score.partition, reference_partition(score.action))

    def test_negative_target_is_approximation_not_exact(self) -> None:
        self.assertNotIn((12, 10), {reference_partition(a) for a in ACTIONS})
        decision = select_balance(HYPOTHESES, ACTIONS, (12, 10), fixture_predict)
        self.assertFalse(decision.exact)
        self.assertGreater(decision.distance, 0)
        self.assertEqual(decision.selected_action, "A03")

    def test_structural_identity_changes_task_sufficiency(self) -> None:
        hs = ("h1", "h2", "h3", "h4")
        task = {"h1": 0, "h2": 0, "h3": 1, "h4": 1}

        def predictor(h: str, action: str) -> int:
            if action == "A":
                return 0 if h in {"h1", "h2"} else 1
            if action == "B":
                return 0 if h in {"h1", "h3"} else 1
            raise KeyError(action)

        pa = structural_partition(hs, "A", predictor)
        pb = structural_partition(hs, "B", predictor)
        self.assertEqual(tuple(sorted(map(len, pa), reverse=True)), (2, 2))
        self.assertEqual(tuple(sorted(map(len, pb), reverse=True)), (2, 2))
        self.assertNotEqual(set(pa), set(pb))

        sa = build_semantic_state(hs, named_partition=("A", pa), task=task)
        sb = build_semantic_state(hs, named_partition=("B", pb), task=task)
        self.assertEqual(sa.sufficiency_status, SufficiencyStatus.SUFFICIENT)
        self.assertEqual(sb.sufficiency_status, SufficiencyStatus.UNRESOLVED)
        self.assertEqual(sa.relevant_distinctions, ())
        self.assertNotEqual(sb.relevant_distinctions, ())

    def test_no_task_never_emits_sufficient(self) -> None:
        hs = ("h1", "h2")
        state = build_semantic_state(
            hs,
            named_partition=("identity", ({"h1"}, {"h2"})),
            task=None,
        )
        self.assertEqual(state.sufficiency_status, SufficiencyStatus.UNRESOLVED)


class ShadowIntegrationTests(unittest.TestCase):
    @staticmethod
    def obs(value: int) -> Obs:
        return Obs(
            game_id="fixture",
            grids=[np.array([[value]], dtype=np.int8)],
            state="NOT_FINISHED",
            levels_completed=0,
            win_levels=1,
            available_actions=[1, 2, 3, 4],
        )

    def test_shadow_records_feedback_without_changing_next_choice(self) -> None:
        base = AOAArc3Core(max_actions=80)
        sem = SemanticAOAArc3Core(max_actions=80)

        first_base = base.choose(self.obs(0))
        first_sem = sem.choose(self.obs(0))
        self.assertEqual(
            (first_base.action_id, first_base.data),
            (first_sem.action_id, first_sem.data),
        )

        second_base = base.choose(self.obs(1))
        second_sem = sem.choose(self.obs(1))
        self.assertEqual(
            (second_base.action_id, second_base.data),
            (second_sem.action_id, second_sem.data),
        )

        self.assertEqual(len(sem.semantic_shadow.events), 1)
        self.assertEqual(
            len(base.model.transition_records),
            len(sem.model.transition_records),
        )


if __name__ == "__main__":
    unittest.main()
