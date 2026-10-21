from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShadowSemanticMemory:
    """Append-only, decision-neutral semantic ledger."""

    events: list[dict[str, Any]] = field(default_factory=list)
    relations: set[tuple[str, str, str]] = field(default_factory=set)
    parent_version: str = "0" * 16

    def observe_transition(
        self,
        *,
        from_state: str,
        action_id: int,
        to_state: str,
        changed_cells: int,
        level_delta: int,
        game_over: bool,
        win: bool,
        provenance: str = "AOAArc3Core",
    ) -> str:
        event = {
            "from_state": str(from_state),
            "action_id": int(action_id),
            "to_state": str(to_state),
            "changed_cells": int(changed_cells),
            "level_delta": int(level_delta),
            "game_over": bool(game_over),
            "win": bool(win),
            "provenance": str(provenance),
            "parent_version": self.parent_version,
        }
        digest = hashlib.sha256(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        event["version_id"] = digest
        self.events.append(event)
        self.relations.add((str(from_state), f"ACTION{int(action_id)}", str(to_state)))
        self.parent_version = digest
        return digest
