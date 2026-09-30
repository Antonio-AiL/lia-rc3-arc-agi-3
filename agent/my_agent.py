#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AOA COSMOS / ARC-FARO ARC-AGI-3 agent candidate
================================================
Status: RC2 / hardened local experimental candidate.

Design goals:
- no hosted API calls or internet dependencies at inference time;
- compatible with the official ARC-AGI-3 Agent interface when placed as
  `agent/my_agent.py` in the official Kaggle starter;
- learn action effects from transitions instead of hard-coding game IDs;
- preserve informative zero-reward transitions (AOA N=-0 operational rule);
- use bounded exploration, transition memory, click saliency and simple path planning;
- fail safely: only legal actions, ACTION6 coordinates clamped to [0,63].

This file is deliberately self-contained for Kaggle packaging.
"""

import hashlib
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

try:  # Official framework
    from agents.agent import Agent
except Exception:  # pragma: no cover - local standalone fallback
    class Agent:  # type: ignore
        MAX_ACTIONS = 80
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

try:  # Official ARC engine
    from arcengine import FrameData, GameAction, GameState
except Exception:  # pragma: no cover - allows local import without toolkit
    FrameData = Any  # type: ignore

    class _FallbackAction:
        def __init__(self, name: str, action_id: int) -> None:
            self.name = name
            self.value = action_id
            self.data: dict[str, Any] = {}
            self.reasoning: dict[str, Any] | None = None
        def set_data(self, data: dict[str, Any]) -> "_FallbackAction":
            self.data = dict(data)
            return self
        def is_complex(self) -> bool:
            return self.name == "ACTION6"
        def __repr__(self) -> str:
            return self.name

    class GameAction:  # type: ignore
        RESET = _FallbackAction("RESET", 0)
        ACTION1 = _FallbackAction("ACTION1", 1)
        ACTION2 = _FallbackAction("ACTION2", 2)
        ACTION3 = _FallbackAction("ACTION3", 3)
        ACTION4 = _FallbackAction("ACTION4", 4)
        ACTION5 = _FallbackAction("ACTION5", 5)
        ACTION6 = _FallbackAction("ACTION6", 6)
        ACTION7 = _FallbackAction("ACTION7", 7)
        @classmethod
        def from_name(cls, name: str) -> _FallbackAction:
            proto = getattr(cls, name)
            return _FallbackAction(proto.name, proto.value)

    class GameState:  # type: ignore
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"


# ------------------------------ data model ------------------------------

@dataclass(frozen=True)
class Choice:
    action_id: int
    data: dict[str, int] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Obs:
    game_id: str
    grids: list[np.ndarray]
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    full_reset: bool = False

    @property
    def grid(self) -> np.ndarray:
        if not self.grids:
            return np.zeros((1, 1), dtype=np.int8)
        return np.asarray(self.grids[-1])


@dataclass
class ActionStats:
    trials: int = 0
    changes: int = 0
    no_changes: int = 0
    level_gains: int = 0
    game_overs: int = 0
    wins: int = 0
    changed_cells_total: int = 0
    negzero_changes: int = 0

    def utility(self) -> float:
        n = max(1, self.trials)
        gain = self.level_gains / n
        change = self.changes / n
        danger = self.game_overs / n
        win = self.wins / n
        novelty = 1.25 / math.sqrt(self.trials + 1.0)
        # Changed-without-reward is treated as evidence, not failure.
        nminuszero = 0.25 * (self.negzero_changes / n)
        return 5.0 * win + 3.5 * gain + 0.8 * change + nminuszero + novelty - 5.0 * danger


@dataclass
class PosteriorStats:
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0
    noops: int = 0

    def update(self, reward: float, noop: bool = False) -> None:
        r = max(0.0, min(1.0, float(reward)))
        self.alpha += r
        self.beta += 1.0 - r
        self.attempts += 1
        if noop:
            self.noops += 1

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)


@dataclass
class Component:
    color: int
    pixels: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int


@dataclass
class Transition:
    prev_hash: str
    action_id: int
    data_key: tuple[tuple[str, int], ...]
    next_hash: str
    changed_cells: int
    animation_changed_cells: int
    level_delta: int
    game_over: bool
    win: bool
    displacement: Optional[tuple[int, int]] = None


# ------------------------------ perception ------------------------------

def _state_name(x: Any) -> str:
    name = getattr(x, "name", None)
    if name:
        return str(name)
    s = str(x)
    return s.split(".")[-1]


def _action_id(a: Any) -> int:
    if isinstance(a, (int, np.integer)):
        return int(a)
    value = getattr(a, "value", None)
    if isinstance(value, int):
        return value
    name = getattr(a, "name", str(a))
    if name == "RESET":
        return 0
    if str(name).startswith("ACTION"):
        try:
            return int(str(name).replace("ACTION", ""))
        except Exception:
            pass
    return -1


def normalize_frame(latest_frame: Any) -> Obs:
    raw_frames = getattr(latest_frame, "frame", None)
    if raw_frames is None:
        grids = [np.zeros((1, 1), dtype=np.int8)]
    else:
        # ARC-AGI-3 FrameData.frame is one-or-more grids.
        if isinstance(raw_frames, np.ndarray) and raw_frames.ndim == 2:
            grids = [raw_frames]
        else:
            grids = [np.asarray(x) for x in list(raw_frames)]
    available = getattr(latest_frame, "available_actions", None) or []
    ids = sorted({i for i in (_action_id(a) for a in available) if 0 <= i <= 7})
    return Obs(
        game_id=str(getattr(latest_frame, "game_id", "unknown")),
        grids=grids,
        state=_state_name(getattr(latest_frame, "state", "NOT_FINISHED")),
        levels_completed=int(getattr(latest_frame, "levels_completed", 0) or 0),
        win_levels=int(getattr(latest_frame, "win_levels", 0) or 0),
        available_actions=ids,
        full_reset=bool(getattr(latest_frame, "full_reset", False)),
    )


def frame_hash(grid: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(grid))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(tuple(a.shape)).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()[:20]


def background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def components(grid: np.ndarray) -> list[Component]:
    g = np.asarray(grid)
    if g.ndim != 2:
        return []
    bg = background_color(g)
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    out: list[Component] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = [(r, c)]
            seen[r, c] = True
            pix: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                pix.append((rr, cc))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in pix]
            cs = [p[1] for p in pix]
            out.append(Component(
                color=color,
                pixels=tuple(sorted(pix)),
                bbox=(min(rs), min(cs), max(rs), max(cs)),
                centroid=(float(sum(rs))/len(rs), float(sum(cs))/len(cs)),
                area=len(pix),
            ))
    return out


def _relative_shape(comp: Component) -> tuple[tuple[int,int], ...]:
    r0,c0,_,_=comp.bbox
    return tuple(sorted((r-r0,c-c0) for r,c in comp.pixels))


def _best_moved_component(prev: np.ndarray, curr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Return (dr, dc, color, area) for the clearest preserved component move.

    Components must have the same color, area and *relative pixel shape*.  This
    prevents static wall fragments with coincidentally equal area from being
    mistaken for the controlled sprite.
    """
    if prev.shape == curr.shape and np.array_equal(prev, curr):
        return None
    pa = components(prev)
    ca = components(curr)
    # Index by a shape-preserving signature. Repeated singleton fields can create
    # thousands of equivalent components; those are intentionally skipped rather
    # than paying O(n^2) cost or learning an arbitrary false displacement.
    buckets: dict[tuple[int, int, tuple[tuple[int,int], ...]], list[Component]] = defaultdict(list)
    for b in ca:
        buckets[(b.color, b.area, _relative_shape(b))].append(b)
    best: tuple[float, int, int, int, int] | None = None
    for a in pa:
        matches = buckets.get((a.color, a.area, _relative_shape(a)), [])
        if len(matches) > 64:
            continue
        for b in matches:
            dr = int(round(b.centroid[0] - a.centroid[0]))
            dc = int(round(b.centroid[1] - a.centroid[1]))
            if dr == 0 and dc == 0:
                continue
            man=abs(dr)+abs(dc)
            score = man + 0.001*a.area
            cand = (score, dr, dc, a.color, a.area)
            if best is None or cand < best:
                best = cand
    return None if best is None else (best[1], best[2], best[3], best[4])


def changed_cells(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        return int(max(a.size, b.size))
    return int(np.count_nonzero(a != b))


def temporal_activity(grids: Iterable[np.ndarray]) -> np.ndarray:
    """Count per-cell changes inside one ARC-AGI-3 frame stack.

    ARC-AGI-3 can return multiple sequential grids for a single action.  The
    final grid remains the control state, but transient motion can carry useful
    evidence even when the first and last grids are equal.  Shape changes are
    handled conservatively by returning activity over the last grid shape.
    """
    seq = [np.asarray(g) for g in grids]
    if not seq:
        return np.zeros((1, 1), dtype=np.int16)
    last = seq[-1]
    if last.ndim != 2:
        return np.zeros((1, 1), dtype=np.int16)
    activity = np.zeros(last.shape, dtype=np.int16)
    for a, b in zip(seq, seq[1:]):
        if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or b.shape != last.shape:
            continue
        activity += (a != b).astype(np.int16)
    return activity


def temporal_changed_cells(grids: Iterable[np.ndarray]) -> int:
    return int(np.count_nonzero(temporal_activity(grids)))


def temporal_saliency_points(grids: Iterable[np.ndarray], max_points: int = 12) -> list[tuple[int, int]]:
    """Candidate ACTION6 points derived from within-action animation.

    Highest-activity cells are mapped to official display coordinates.  This is
    deliberately generic: no game IDs or semantic labels are used.
    """
    seq = [np.asarray(g) for g in grids]
    if not seq:
        return []
    g = seq[-1]
    if g.ndim != 2:
        return []
    act = temporal_activity(seq)
    if not np.any(act):
        return []
    h, w = g.shape
    coords = np.argwhere(act > 0)
    coords = sorted(coords.tolist(), key=lambda rc: (-int(act[rc[0], rc[1]]), rc[0], rc[1]))
    pts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in coords:
        x = int(round((c + 0.5) * 64.0 / max(1, w) - 0.5))
        y = int(round((r + 0.5) * 64.0 / max(1, h) - 0.5))
        pt = (min(63, max(0, x)), min(63, max(0, y)))
        if pt not in seen:
            seen.add(pt)
            pts.append(pt)
        if len(pts) >= max_points:
            break
    return pts


def saliency_points(grid: np.ndarray, max_points: int = 20) -> list[tuple[int, int]]:
    """Candidate click coordinates in display space [0,63]x[0,63]."""
    g = np.asarray(grid)
    h, w = g.shape
    comps = components(g)
    # Rare colors / small components first, then larger components.
    color_area = Counter()
    for comp in comps:
        color_area[comp.color] += comp.area
    comps.sort(key=lambda x: (color_area[x.color], x.area, x.color, x.centroid))
    pts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for comp in comps:
        r, c = comp.centroid
        # Official click coordinates use display coordinates 0..63.
        x = int(round((c + 0.5) * 64.0 / max(1, w) - 0.5))
        y = int(round((r + 0.5) * 64.0 / max(1, h) - 0.5))
        p = (min(63, max(0, x)), min(63, max(0, y)))
        if p not in seen:
            seen.add(p)
            pts.append(p)
            if len(pts) >= max_points:
                break
    # Center is a low-priority generic probe.
    center = (32, 32)
    if center not in seen and len(pts) < max_points:
        pts.append(center)
    return pts


def grid_cell_from_display(grid: np.ndarray, x: int, y: int) -> tuple[int, int]:
    h, w = grid.shape
    c = min(w - 1, max(0, int((x + 0.5) * w / 64.0)))
    r = min(h - 1, max(0, int((y + 0.5) * h / 64.0)))
    return r, c


# ------------------------------ world model ------------------------------

class WorldModel:
    def __init__(self) -> None:
        self.stats: dict[int, ActionStats] = defaultdict(ActionStats)
        self.transitions: dict[str, dict[tuple[int, tuple[tuple[str,int], ...]], str]] = defaultdict(dict)
        self.transition_records: list[Transition] = []
        self.action_disp: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
        self.avatar_signature: tuple[int, int] | None = None  # color, area
        self.success_sequences: list[list[Choice]] = []
        self.click_effects: dict[tuple[int, int], ActionStats] = defaultdict(ActionStats)
        self.seen_state_counts: Counter[str] = Counter()

    def observe(self, prev: Obs, action: Choice, curr: Obs) -> Transition:
        p = prev.grid
        c = curr.grid
        ph = frame_hash(p)
        ch = frame_hash(c)
        delta = changed_cells(p, c)
        animation_delta = temporal_changed_cells(curr.grids)
        level_delta = max(0, curr.levels_completed - prev.levels_completed)
        game_over = curr.state == "GAME_OVER"
        win = curr.state == "WIN"
        displacement = None
        # Do not learn a movement model across level boundaries: the new level
        # can redraw unrelated objects and create false "teleports".
        moved = None if (level_delta > 0 or win or game_over) else _best_moved_component(p, c)
        if moved is not None and action.action_id in (1,2,3,4):
            dr, dc, color, area = moved
            displacement = (dr, dc)
            self.action_disp[action.action_id][displacement] += 1
            self.avatar_signature = (color, area)

        st = self.stats[action.action_id]
        st.trials += 1
        if delta > 0 or animation_delta > 0 or p.shape != c.shape:
            st.changes += 1
            st.changed_cells_total += max(delta, animation_delta)
            if level_delta == 0 and not game_over and not win:
                st.negzero_changes += 1
        else:
            st.no_changes += 1
        st.level_gains += level_delta
        st.game_overs += int(game_over)
        st.wins += int(win)

        if action.action_id == 6 and action.data:
            pt = (int(action.data.get("x", 0)), int(action.data.get("y", 0)))
            cs = self.click_effects[pt]
            cs.trials += 1
            informative_change = (delta > 0 or animation_delta > 0)
            cs.changes += int(informative_change)
            cs.no_changes += int(not informative_change)
            cs.level_gains += level_delta
            cs.game_overs += int(game_over)
            cs.wins += int(win)
            if informative_change and level_delta == 0 and not game_over:
                cs.negzero_changes += 1

        data_key = tuple(sorted((str(k), int(v)) for k, v in action.data.items()))
        self.transitions[ph][(action.action_id, data_key)] = ch
        tr = Transition(
            prev_hash=ph,
            action_id=action.action_id,
            data_key=data_key,
            next_hash=ch,
            changed_cells=delta,
            animation_changed_cells=animation_delta,
            level_delta=level_delta,
            game_over=game_over,
            win=win,
            displacement=displacement,
        )
        self.transition_records.append(tr)
        self.seen_state_counts[ch] += 1
        return tr

    def dominant_displacement(self, action_id: int) -> Optional[tuple[int, int]]:
        c = self.action_disp.get(action_id)
        if not c:
            return None
        return c.most_common(1)[0][0]

    def action_utility(self, action_id: int) -> float:
        # ACTION7 (undo) is a recovery operator, not a normal exploratory action.
        penalty = -0.5 if action_id == 7 else 0.0
        return self.stats[action_id].utility() + penalty


# ------------------------------ planning helpers ------------------------------

DEFAULT_DIRECTION_HINTS: dict[int, tuple[int,int]] = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


def _avatar_component(grid: np.ndarray, model: WorldModel) -> Optional[Component]:
    comps = components(grid)
    if not comps:
        return None
    if model.avatar_signature is not None:
        color, area = model.avatar_signature
        matches = [c for c in comps if c.color == color and c.area == area]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None
    # Conservative generic guess: a unique, small component of a rare color.
    color_counts = Counter(c.color for c in comps)
    candidates = [c for c in comps if color_counts[c.color] == 1 and c.area <= 9]
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c.area, c.color))


def _candidate_targets(grid: np.ndarray, avatar: Component) -> list[Component]:
    comps = components(grid)
    others = [c for c in comps if not (c.color == avatar.color and c.pixels == avatar.pixels)]
    if not others:
        return []
    # Goal-like objects are usually compact and visually distinctive.  Avoid
    # treating fragments of a high-mass wall color as targets merely because a
    # connected fragment happens to be small.
    total_by_color: Counter[int] = Counter()
    count_by_color: Counter[int] = Counter()
    for c in comps:
        total_by_color[c.color] += c.area
        count_by_color[c.color] += 1
    compact = [
        c for c in others
        if total_by_color[c.color] <= 12 and c.color != avatar.color
    ]
    pool = compact if compact else others
    pool.sort(key=lambda c: (
        count_by_color[c.color] != 1,
        total_by_color[c.color],
        c.area,
        c.color,
    ))
    return pool[:4]


def _direction_map(model: WorldModel, available: Iterable[int]) -> dict[int, tuple[int,int]]:
    out: dict[int, tuple[int,int]] = {}
    for aid in available:
        if aid not in (1,2,3,4):
            continue
        learned = model.dominant_displacement(aid)
        out[aid] = learned if learned is not None else DEFAULT_DIRECTION_HINTS[aid]
    return out


def _plan_toward_target(grid: np.ndarray, model: WorldModel, available: list[int]) -> Optional[int]:
    """Small BFS planner over inferred avatar/target geometry.

    It does not assume a specific game ID. It uses learned action displacements when
    available, falling back to the official directional semantics of ACTION1..4.
    """
    avatar = _avatar_component(grid, model)
    if avatar is None:
        return None
    direction_map = _direction_map(model, available)
    if len(direction_map) < 2:
        return None

    targets = _candidate_targets(grid, avatar)
    if not targets:
        return None

    h, w = grid.shape
    ar = int(round(avatar.centroid[0]))
    ac = int(round(avatar.centroid[1]))
    # Treat background as traversable; target cells are also traversable.
    bg = background_color(grid)
    target_cells = set()
    for t in targets[:4]:
        target_cells.update(t.pixels)
    if not target_cells:
        return None

    q = deque([(ar, ac)])
    parent: dict[tuple[int,int], tuple[tuple[int,int], int] | None] = {(ar,ac): None}
    found: tuple[int,int] | None = None
    while q and len(parent) <= 4096:
        r, c = q.popleft()
        if (r, c) in target_cells and (r, c) != (ar, ac):
            found = (r, c)
            break
        for aid, (dr, dc) in direction_map.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in parent:
                continue
            cell = int(grid[nr, nc])
            if cell != bg and (nr, nc) not in target_cells:
                continue
            parent[(nr,nc)] = ((r,c), aid)
            q.append((nr,nc))
    if found is None:
        return None
    cur = found
    first_action: int | None = None
    while parent[cur] is not None:
        prev, aid = parent[cur]  # type: ignore[misc]
        first_action = aid
        cur = prev
    return first_action


# ------------------------------ controller ------------------------------

class AOAArc3Core:
    """Offline transition-learning controller for ARC-AGI-3."""

    def __init__(self, max_actions: int = 80) -> None:
        self.max_actions = int(max_actions)
        self.model = WorldModel()
        self.prev_obs: Obs | None = None
        self.prev_choice: Choice | None = None
        self.actions_taken = 0
        self.level_start_index = 0
        self.episode_choices: list[Choice] = []
        self.state_attempts: dict[str, Counter[tuple[int, tuple[tuple[str,int],...]]]] = defaultdict(Counter)
        self.clicked_points_by_state: dict[str, set[tuple[int,int]]] = defaultdict(set)
        self.last_transition: Transition | None = None
        self.stagnation = 0
        # RC3 donor: state-conditioned Thompson posterior + bounded evidence ledger.
        self.rng = random.Random(314159)
        self.posterior: dict[tuple[str, int, tuple[tuple[str, int], ...]], PosteriorStats] = defaultdict(PosteriorStats)
        self.recent_edge_keys: deque[tuple[str, int, tuple[tuple[str, int], ...]]] = deque(maxlen=12)
        self.faro_entries: deque[dict[str, Any]] = deque(maxlen=512)
        self.faro_prev = "0" * 64

    def reset_local(self, keep_model: bool = True) -> None:
        model = self.model if keep_model else WorldModel()
        self.__init__(self.max_actions)
        self.model = model

    def _edge_key(self, state_hash: str, choice: Choice) -> tuple[str, int, tuple[tuple[str, int], ...]]:
        data_key = tuple(sorted((str(k), int(v)) for k, v in choice.data.items()))
        return (state_hash, int(choice.action_id), data_key)

    def _posterior_score(self, key: tuple[str, int, tuple[tuple[str, int], ...]]) -> float:
        st = self.posterior[key]
        sample = st.sample(self.rng)
        novelty = 0.55 / (1.0 + st.attempts)
        noop_penalty = 0.18 * st.noops
        return sample + novelty - noop_penalty

    def _faro_append(self, event: dict[str, Any]) -> None:
        payload = {"prev": self.faro_prev, **event}
        digest = hashlib.sha256((self.faro_prev + repr(sorted(payload.items(), key=lambda kv: kv[0]))).encode("utf-8")).hexdigest()
        self.faro_entries.append({"prev": self.faro_prev, "hash": digest, "event": event})
        self.faro_prev = digest

    def _process_feedback(self, obs: Obs) -> None:
        if self.prev_obs is None or self.prev_choice is None:
            return
        tr = self.model.observe(self.prev_obs, self.prev_choice, obs)
        self.last_transition = tr

        # RC3 donor: state-conditioned credit. External progress dominates;
        # N=-0 preserves any real visual/animation change with small positive credit.
        key = self._edge_key(tr.prev_hash, self.prev_choice)
        changed = max(tr.changed_cells, tr.animation_changed_cells)
        total = max(1, int(self.prev_obs.grid.size), int(obs.grid.size))
        reward = 0.0
        if changed > 0:
            reward = max(0.04, min(0.45, (changed / total) * 3.0))
        if tr.level_delta > 0 or tr.win:
            reward = 1.0
        if tr.game_over:
            reward = 0.0
        self.posterior[key].update(reward, noop=(changed == 0 and tr.level_delta <= 0 and not tr.win))
        self.recent_edge_keys.append(key)
        self._faro_append({
            "from": tr.prev_hash, "action": tr.action_id, "data": dict(tr.data_key),
            "to": tr.next_hash, "changed": tr.changed_cells,
            "animation_changed": tr.animation_changed_cells,
            "level_delta": tr.level_delta, "game_over": tr.game_over, "win": tr.win,
        })
        if tr.level_delta > 0 or tr.win:
            for depth, ek in enumerate(reversed(self.recent_edge_keys)):
                self.posterior[ek].update(max(0.10, 0.75 / (1 + depth)), noop=False)

        if tr.level_delta > 0 or tr.win:
            seq = self.episode_choices[self.level_start_index:]
            if seq:
                self.model.success_sequences.append(list(seq))
            self.level_start_index = len(self.episode_choices)
            self.stagnation = 0
        elif (tr.changed_cells > 0 or tr.animation_changed_cells > 0) and not tr.game_over:
            # N=-0: no immediate reward, but state/animation changed -> preserve as informative.
            self.stagnation = max(0, self.stagnation - 1)
        elif tr.game_over:
            self.stagnation += 2
        else:
            self.stagnation += 1

    def choose(self, obs: Obs) -> Choice:
        self._process_feedback(obs)
        state_hash = frame_hash(obs.grid)
        self.model.seen_state_counts[state_hash] += 1

        if obs.state in ("NOT_PLAYED", "GAME_OVER") or (obs.full_reset and self.actions_taken == 0):
            choice = Choice(0, {}, "start-or-recover")
            self._commit(obs, choice)
            return choice

        legal = [a for a in obs.available_actions if 1 <= a <= 7]
        if not legal:
            # RESET is the only safe fallback if the environment exposes nothing else.
            choice = Choice(0, {}, "no-legal-action-fallback")
            self._commit(obs, choice)
            return choice

        # 1) Reuse a short successful prefix when the current action remains legal.
        replay = self._replay_candidate(legal)
        if replay is not None:
            self._commit(obs, replay)
            return replay

        # 2) Click-only or click-dominant environments: inspect salient components.
        click_choice = self._click_candidate(obs, legal, state_hash)
        if click_choice is not None and (legal == [6] or self.model.stats[6].level_gains > 0):
            self._commit(obs, click_choice)
            return click_choice

        # 3) Geometry-aware navigation after or while learning directional effects.
        move = _plan_toward_target(obs.grid, self.model, legal)
        if move is not None:
            key = (move, tuple())
            tries = self.state_attempts[state_hash][key]
            if tries < 2 and self.model.stats[move].game_overs == 0:
                choice = Choice(move, {}, "geometry-plan")
                self._commit(obs, choice)
                return choice

        # 4) Probe untried simple actions first. This is active exploration.
        simple = [a for a in legal if a != 6 and a != 7]
        untried_here = []
        for aid in simple:
            key = (aid, tuple())
            if self.state_attempts[state_hash][key] == 0:
                untried_here.append(aid)
        if untried_here:
            # Global evidence breaks ties: safer / more useful first.
            aid = max(untried_here, key=lambda x: (self.model.action_utility(x), -x))
            choice = Choice(aid, {}, "active-exploration")
            self._commit(obs, choice)
            return choice

        # 5) Try a new salient click if ACTION6 is legal.
        if click_choice is not None:
            self._commit(obs, click_choice)
            return click_choice

        # 6) If stalled and undo exists, backtrack once rather than thrash.
        if 7 in legal and self.stagnation >= 3:
            key = (7, tuple())
            if self.state_attempts[state_hash][key] == 0:
                choice = Choice(7, {}, "stagnation-undo")
                self._commit(obs, choice)
                return choice

        # 7) Evidence-weighted exploitation with per-state repetition penalty.
        scored: list[tuple[float, int]] = []
        for aid in simple or [a for a in legal if a != 7]:
            key = (aid, tuple())
            repeats = self.state_attempts[state_hash][key]
            posterior_key = (state_hash, aid, tuple())
            score = self.model.action_utility(aid) - 0.7 * repeats
            score += 0.65 * self._posterior_score(posterior_key)
            # Avoid actions already strongly associated with game over.
            st = self.model.stats[aid]
            if st.game_overs >= 1 and st.game_overs >= st.level_gains + st.wins:
                score -= 4.0
            scored.append((score, aid))
        if scored:
            scored.sort(key=lambda x: (-x[0], x[1]))
            choice = Choice(scored[0][1], {}, "evidence-exploitation")
            self._commit(obs, choice)
            return choice

        aid = legal[0]
        if aid == 6:
            choice = Choice(6, {"x": 32, "y": 32}, "last-legal-click-fallback")
        else:
            choice = Choice(aid, {}, "last-legal-fallback")
        self._commit(obs, choice)
        return choice

    def _replay_candidate(self, legal: list[int]) -> Choice | None:
        if not self.model.success_sequences:
            return None
        # only replay very short, repeatedly-successful prefixes; avoid click coordinates
        counts: Counter[tuple[int,...]] = Counter()
        for seq in self.model.success_sequences:
            ids = tuple(c.action_id for c in seq[:4] if c.action_id != 0)
            if ids:
                counts[ids] += 1
        if not counts:
            return None
        prefix, support = counts.most_common(1)[0]
        if support < 2:
            return None
        level_pos = len(self.episode_choices) - self.level_start_index
        if level_pos < len(prefix):
            aid = prefix[level_pos]
            if aid in legal and aid != 6:
                return Choice(aid, {}, f"replay-success-prefix/support={support}")
        return None

    @staticmethod
    def _fallback_click_points(limit: int = 40) -> list[tuple[int, int]]:
        """Deterministic low-discrepancy ACTION6 probes in display space.

        Used only after visual/temporal saliency points for a state are exhausted.
        This keeps click-only environments legal and gives blank/ambiguous screens
        bounded spatial coverage without relying on randomness or game IDs.
        """
        def vdc(n: int, base: int) -> float:
            v, denom = 0.0, 1.0
            while n:
                n, rem = divmod(n, base)
                denom *= base
                v += rem / denom
            return v
        out: list[tuple[int, int]] = []
        for i in range(1, limit + 1):
            x = min(63, max(0, int(vdc(i, 2) * 64)))
            y = min(63, max(0, int(vdc(i, 3) * 64)))
            pt = (x, y)
            if pt not in out:
                out.append(pt)
        return out

    def _click_candidate(self, obs: Obs, legal: list[int], state_hash: str) -> Choice | None:
        if 6 not in legal:
            return None
        points: list[tuple[int, int]] = []
        for pt in temporal_saliency_points(obs.grids) + saliency_points(obs.grid):
            if pt not in points:
                points.append(pt)
        used = self.clicked_points_by_state[state_hash]
        ranked: list[tuple[float, tuple[int, int]]] = []
        for pt in points:
            if pt in used:
                continue
            st = self.model.click_effects.get(pt, ActionStats())
            pk = (state_hash, 6, (("x", int(pt[0])), ("y", int(pt[1]))))
            ranked.append((st.utility() + 0.40 * self._posterior_score(pk), pt))
        if ranked:
            ranked.sort(key=lambda x: (-x[0], x[1]))
            x, y = ranked[0][1]
            return Choice(6, {"x": int(x), "y": int(y)}, "salient-click-probe")

        # Saliency exhausted. Probe the display deterministically before repeating.
        for x, y in self._fallback_click_points():
            if (x, y) not in used:
                return Choice(6, {"x": x, "y": y}, "coverage-click-probe")

        # If the state survives more clicks than our bounded coverage budget, repeat
        # the globally safest/effective known coordinate. Never emit bare ACTION6.
        candidates = points + self._fallback_click_points()
        if not candidates:
            candidates = [(32, 32)]
        best = max(
            candidates,
            key=lambda pt: (self.model.click_effects.get(pt, ActionStats()).utility(), -pt[0], -pt[1]),
        )
        return Choice(6, {"x": int(best[0]), "y": int(best[1])}, "safe-click-repeat")

    def _commit(self, obs: Obs, choice: Choice) -> None:
        aid = int(choice.action_id)
        data = {k: min(63, max(0, int(v))) for k, v in choice.data.items()}
        choice = Choice(aid, data, choice.reason)
        sh = frame_hash(obs.grid)
        key = (aid, tuple(sorted(data.items())))
        self.state_attempts[sh][key] += 1
        if aid == 6 and data:
            self.clicked_points_by_state[sh].add((data.get("x", 0), data.get("y", 0)))
        self.prev_obs = obs
        self.prev_choice = choice
        self.episode_choices.append(choice)
        self.actions_taken += 1


# ------------------------------ official adapter ------------------------------

class MyAgent(Agent):
    """Drop-in class for the official ARC-AGI-3 Kaggle starter."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.core = AOAArc3Core(max_actions=self.MAX_ACTIONS)

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return _state_name(getattr(latest_frame, "state", "")) == "WIN"

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        obs = normalize_frame(latest_frame)
        choice = self.core.choose(obs)
        data = {str(k): min(63, max(0, int(v))) for k, v in choice.data.items()}
        if choice.action_id == 6:
            # Competition/local runtimes expect both coordinates for ACTION6.
            data.setdefault("x", 32)
            data.setdefault("y", 32)
        if choice.action_id == 0:
            action = GameAction.RESET
        else:
            action = GameAction.from_name(f"ACTION{choice.action_id}")
        if hasattr(action, "set_data"):
            action.set_data(data)
        # Official recorder can retain concise evidence without exposing private CoT.
        try:
            action.reasoning = {
                "agent": "AOA_ARC3_RC3_HYBRID",
                "reason": choice.reason,
                "levels_completed": obs.levels_completed,
                "state_hash": frame_hash(obs.grid),
                "actions_taken": self.core.actions_taken,
                "faro_events": len(self.core.faro_entries),
            }
        except Exception:
            pass
        return action


__all__ = [
    "MyAgent",
    "AOAArc3Core",
    "Obs",
    "Choice",
    "WorldModel",
    "normalize_frame",
    "frame_hash",
    "components",
    "saliency_points",
    "temporal_activity",
    "temporal_changed_cells",
    "temporal_saliency_points",
]
