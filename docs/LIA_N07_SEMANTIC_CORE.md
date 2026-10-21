# LIA-N07-SEM · Semantic core (experimental)

Status: **SHADOW / not integrated into stable RC3**.

This contribution adds the smallest falsifiable Node 07 module without moving or
rewriting the published competition agent.

## Claim under test

A partition's numeric shape is not sufficient to preserve task-relevant meaning.

Two experiments may satisfy:

`lambda(A) == lambda(B)`

while their structural partitions differ:

`Pi(A) != Pi(B)`

and only one refines the task partition.

## T03 REVERSE-BALANCE

The deterministic fixture contains 22 hypotheses. `A03` and `A07` both
generate `(11, 11)` but are structurally different. The selector must:

1. keep both exact solutions;
2. choose `A03` only through canonical tie-breaking;
3. mark `(12, 10)` as an approximation when no exact action exists;
4. agree with an independent oracle implemented in the tests.

## SHADOW integration

`agent/my_agent.py` remains untouched.

`agent/my_agent_semantic.py` subclasses the RC3 controller and appends transition
relations to `ShadowSemanticMemory`. The semantic ledger is not consulted by
`choose`, so it cannot alter policy by design. A test checks decision equality
between baseline and semantic-shadow controllers on the same feedback sequence.

## Run

Desktop/Linux:

```bash
python -m unittest discover -s tests -p 'test_lia_n07_semantic.py' -v
```

Termux:

```bash
bash scripts/termux/run_lia_n07_t03.sh
```

## Acceptance boundary

Passing these tests does **not** prove ARC-AGI-3 improvement. It establishes only
the T03 mechanics, task-sufficiency distinction, and decision-neutral SHADOW
integration. Independent review by Node 19 remains required before promotion.
