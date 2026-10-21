#!/usr/bin/env python3
"""Experimental SHADOW adapter for LIA-N07-SEM.

The published RC3 agent in `agent/my_agent.py` is intentionally left untouched.
This candidate subclasses the same controller and only appends semantic evidence
after feedback. It does not read semantic memory during action selection.
"""

from __future__ import annotations

from typing import Any

from agent.my_agent import AOAArc3Core, MyAgent as BaselineMyAgent
from lia_core.semantic import ShadowSemanticMemory


class SemanticAOAArc3Core(AOAArc3Core):
    def __init__(self, max_actions: int = 80) -> None:
        super().__init__(max_actions=max_actions)
        self.semantic_shadow = ShadowSemanticMemory()

    def _process_feedback(self, obs: Any) -> None:
        before = len(self.model.transition_records)
        super()._process_feedback(obs)
        if len(self.model.transition_records) == before:
            return
        tr = self.model.transition_records[-1]
        self.semantic_shadow.observe_transition(
            from_state=tr.prev_hash,
            action_id=tr.action_id,
            to_state=tr.next_hash,
            changed_cells=max(tr.changed_cells, tr.animation_changed_cells),
            level_delta=tr.level_delta,
            game_over=tr.game_over,
            win=tr.win,
            provenance="agent/my_agent_semantic.py",
        )


class MyAgent(BaselineMyAgent):
    """Drop-in experimental agent with decision-neutral semantic shadowing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.core = SemanticAOAArc3Core(max_actions=self.MAX_ACTIONS)


__all__ = ["MyAgent", "SemanticAOAArc3Core"]
