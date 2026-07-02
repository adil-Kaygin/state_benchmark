# Issue 15: Free-running curriculum re-runs the full network on every growing prefix — O(T²)–O(T³) work (Transformer & Mamba)

**Severity: high, currently dormant.** This is a wall-clock defect: whenever the
Issue-13 exposure-bias fine-tune is switched on it makes each curriculum epoch
~100–200× slower than a teacher-forced epoch, and for Mamba it ignores an O(T)
recurrence the model already implements. It does not bite the production 30-06
config today because that config leaves `curriculum_epochs` unset (→ 0) for
Transformer and Mamba, so the free-running loop never runs at scale — but that
also means the exposure-bias mitigation Issue 13 documents is effectively
disabled in production. The moment anyone turns the curriculum on at a realistic
T, this is the reason it will be unusably slow (not an OOM in practice — the
retained graph inflates allocator pressure and step time, not a hard crash on
the hardware in use).

## Problem Description

The shared free-running forward
([`_neural_base.py:174-233`](../estimators/neural/_neural_base.py#L174-L233)),
used by the Transformer and Mamba during the trailing `curriculum_epochs`
(Issue 13), is implemented as: at every timestep, append one feature row,
re-stack the entire prefix, and run the **whole network forward over the whole
prefix** ([`_neural_base.py:226-229`](../estimators/neural/_neural_base.py#L226-L229)),
keeping every one of those T forward passes alive in a single autograd graph
(the loss consumes the last position of each pass).

Three compounding defects:

1. **Quadratic-to-cubic recompute.** For a batch of length-T trajectories the
   loop performs T full-sequence passes over prefixes of length 1..T. Token
   work is T(T+1)/2 ≈ 200× a single teacher-forced pass at T=400. For the
   Transformer the attention cost inside those passes sums to O(T³) — ~134×
   the FLOPs of one full causal pass. The validation pass repeats the same
   sequential loop every curriculum epoch.

2. **Full graph retention inflates every step.** Because the fed-back state must
   carry gradient, none of the T prefix passes can be freed until backward, so
   the retained autograd graph scales as ~200× a single pass. This shows up as
   allocator pressure and step-time inflation (not necessarily a hard OOM on the
   hardware in use), and it scales with every knob the configs raise — T,
   width, layers, batch. It is invisible today only because the curriculum runs
   solely in the 01-07 smoke config at T=50–100.

3. **Mamba discards its own O(T) recurrence.** The free-running loop calls the
   network's parallel-scan `forward` on each prefix, so Mamba pays O(T²) scan
   work *plus* the scan's heavy log-space intermediates (see Issue 17) once per
   prefix — even though `_MambaBlock.step`
   ([`mamba.py:134-159`](../estimators/neural/mamba.py#L134-L159)) already
   implements the exact same math as a constant-state O(1)-per-step recurrence.
   A free-running Mamba pass should cost the same order as KalmanNet's Phase-2
   loop, not T/2 times more.

Minor adjunct: during the curriculum tail, the Issue-9 teacher-forced feature
cache is still sliced, shipped to the device, and then ignored on every batch
([`_neural_base.py:412-414`](../estimators/neural/_neural_base.py#L412-L414)).

## Impact Analysis

- **Training speed.** A curriculum epoch costs two orders of magnitude more
  than a teacher-forced epoch. The whole point of the Transformer/Mamba
  training design (documented in Issue 13 and the estimator docstrings) is that
  the warm-start is parallel and the fine-tune is "the irreducible sequential
  cost already accepted for KalmanNet Phase 2" — i.e. O(T) network steps per
  trajectory. The current implementation is not that cost; it is T/2 times that
  cost, with the Transformer's attention adding another factor of T/3.
- **The mitigation is unusable in practice.** Because the fine-tune is this slow
  at any realistic T, it stays switched off at production scale (30-06 sets
  `curriculum_epochs=0` for both models), so the exposure-bias gap Issue 13
  documents as "not apples-to-apples" is never actually closed. Combined with
  Issue 16 (the fine-tuned weights are discarded by the checkpoint logic even
  when the loop does run), turning the curriculum on today buys nothing but a
  large wall-clock bill.
- **Scientific soundness.** If the curriculum is silently configured off (or
  down to T=50 smoke lengths) to dodge the cost, the exposure-bias gap that
  Issue 13 documents as "not apples-to-apples" quietly returns.

## Refactoring Strategy

1. **Mamba — drive the fine-tune through the recurrent step path.** Replace the
   prefix-rerun with a single sequential loop that carries the per-block
   convolution FIFO and SSM state exactly as `_estimate_sequential_cpu` does,
   but batched and differentiable. One structural prerequisite: the step path's
   in-place FIFO update (roll plus index-assign) must be made functional
   (out-of-place concatenation of the shifted window and the new column) so
   autograd can traverse the carried state. Cost becomes O(T) network steps per
   batch — identical in shape to KalmanNet Phase 2 — and the per-step state is
   constant-size, so retention is linear in T. Like KalmanNet Phase 2 (Issue
   12), this launch-bound loop should default to the CPU, with the explicit
   device kwarg as the escape hatch.

2. **Transformer — two-pass scheduled-sampling instead of a differentiable
   prefix rerun.** Pass one runs the sequential generation **without gradient**:
   feed back the model's own previous estimate step by step, exactly as the CPU
   inference path does, purely to *materialize the self-generated input
   features* (states, predictions, innovations). Pass two runs the existing
   single parallel causally-masked forward over those frozen features and takes
   the loss there. Gradient then stops at the fed-back state — the standard
   scheduled-sampling construction — turning retention back into exactly one
   parallel pass while the inputs match the deployment distribution. If
   gradient through the feedback is deemed essential, cap it instead with short
   truncated free-running windows seeded from ground-truth prefixes; never
   retain T full-sequence graphs.

3. **Shared loop hygiene.** Whatever forward replaces the current one must stop
   re-stacking the full feature prefix every step (append into a preallocated
   buffer or slice a growing view once per step), and `fit()` should skip
   carrying/transferring the teacher-forced feature cache on batches consumed
   by the free-running loss.

4. **Acceptance check.** On the profiling notebook's reduced Lorenz config, a
   curriculum epoch must land within a small constant factor (not a factor of
   T) of a teacher-forced epoch for Mamba, and within the cost of one no-grad
   sequential generation plus one parallel pass for the Transformer; peak
   memory during the curriculum tail must be the same order as Phase-1 epochs.

## Out of scope

- The checkpoint/scheduler/early-stopping interaction of the curriculum — that
  is Issue 16 and must be fixed regardless of the forward-pass rewrite.
- KalmanNet, PINN, Neural-ODE (already O(T) free-running).
- The teacher-forced caches (Issue 9) — unchanged and still valid for Phase 1.
