# Issue 16: Best-checkpoint / scheduler / early-stopping bookkeeping spans the teacher-forced→free-running objective switch — the exposure-bias fine-tune is silently discarded (Transformer & Mamba)

**Severity: critical — a silent-failure correctness bug.** The Issue-13
curriculum runs, burns its (enormous, see Issue 15) compute, and then `fit()`
loads the *teacher-forced* weights anyway. The deployed model is exactly as
exposure-biased as before the feature existed, while the benchmark records the
mitigation as active.

## Problem Description

`SequentialNeuralFilter.fit()` runs one flat epoch loop with a single
best-validation tracker, one scheduler, and one early-stopping counter for the
whole fit ([`_neural_base.py:394-478`](../estimators/neural/_neural_base.py#L394-L478)).
The Transformer/Mamba curriculum (Issue 13) flips the training **objective**
mid-loop via the epoch hook: the last `curriculum_epochs` epochs compute both
the train and the validation loss **free-running** instead of teacher-forced
([`transformer.py:200-219`](../estimators/neural/transformer.py#L200-L219),
[`mamba.py:325-344`](../estimators/neural/mamba.py#L325-L344)).

Teacher-forced and free-running validation MSE are not on the same scale. The
profiling run measured the regime gap directly on KalmanNet: teacher-forced val
1.21 versus free-running val 14.0 on the same data
([`brain_storm/01_07.md` §6](../brain_storm/01_07.md)) — an ~12× jump by
construction, not by training failure. Four consequences follow:

1. **The fine-tuned weights never win the checkpoint.** The free-running
   epochs' val losses are compared against the incomparably lower teacher-forced
   minimum ([`_neural_base.py:448-455`](../estimators/neural/_neural_base.py#L448-L455)).
   `improved` stays false for the entire curriculum tail, and at the end of
   `fit()` the best state dict — from a teacher-forced epoch — is loaded back
   ([`_neural_base.py:476-478`](../estimators/neural/_neural_base.py#L476-L478)).
   The entire fine-tune is thrown away, silently, every time.

2. **Early stopping can skip or truncate the curriculum.** `epochs_no_improve`
   accumulates across the regime boundary. With a plateaued teacher-forced
   phase, patience can fire *before* the tail begins (the curriculum never
   runs at all), and once inside the tail every epoch counts as
   "no improvement," so patience fires after at most that many fine-tune
   epochs regardless of whether the free-running loss is still falling.

3. **The plateau scheduler is poisoned at the handoff.** The regime jump reads
   as sustained non-improvement, so the LR is cut exactly when the optimizer
   starts on a new objective it has never minimized.

4. **The training history is unlabeled across regimes.** `history_` mixes the
   two objectives in one `val_loss` series with no phase column, so the loss
   plots show an inexplicable 10×+ jump. KalmanNet solved every one of these
   problems with explicit per-phase bookkeeping — fresh optimizer, scheduler,
   best-checkpoint and patience per phase, phase-tagged history
   ([`kalmannet.py:753-809`](../estimators/neural/kalmannet.py#L753-L809)) —
   and the Issue-13 spec said to "directly reuse the KalmanNet pattern." The
   implementation instead reused only the epoch-flag flip.

## Impact Analysis

- **Convergence / scientific validity.** The benchmark believes Transformer and
  Mamba train with an exposure-bias mitigation; the artifact that reaches
  `estimate()` is the teacher-forced network. Every conclusion drawn from
  "curriculum on" runs — including any before/after RMSE comparison demanded by
  Issue 13's acceptance criteria — is comparing a model to itself. This is
  precisely the class of silent-failure the repository's fail-fast rule exists
  to prevent: an invalid configuration producing a plausible-looking result.
- **Wasted compute at the worst possible price.** Per Issue 15 the curriculum
  epochs are ~two orders of magnitude more expensive than teacher-forced
  epochs; 100% of that spend currently buys nothing.
- **Compounding with tuning decisions.** A practitioner seeing "val loss
  exploded in the last N epochs and the best epoch is earlier" will naturally
  shrink or disable `curriculum_epochs`, entrenching the bias the feature was
  built to remove.

## Refactoring Strategy

1. **Restructure the curriculum as two explicit phases,** mirroring KalmanNet's
   per-phase runner: the teacher-forced phase owns its optimizer/scheduler
   state, best-checkpoint, and patience counter; on completion, load that
   phase's best weights; then start the free-running phase with a fresh
   scheduler, a fresh best-value initialized to infinity, and a fresh patience
   counter. The free-running phase — the deployed objective — owns the final
   checkpoint that `fit()` loads. Move the phase decision out of the
   per-epoch hook and into the fit structure so an objective switch can never
   again share a comparison baseline across regimes.
2. **Tag history rows with the phase,** as KalmanNet's history already does, so
   the regime jump in the plots is attributable and each phase's convergence is
   independently inspectable.
3. **Keep the epoch semantics explicit and consistent.** Today the same knob
   name means opposite things: KalmanNet's `curriculum_epochs` *adds*
   teacher-forced epochs before `num_epochs` free-running ones, while
   Transformer/Mamba *convert the last* `curriculum_epochs` of `num_epochs` to
   free-running. Align on the KalmanNet semantics (total = warm-start epochs +
   deployed-objective epochs) during the restructure, and validate at
   construction time that the resulting phase budgets are both positive —
   fail fast on a configuration where one phase is silently empty.
4. **Acceptance check.** After a curriculum fit, the loaded weights must come
   from a free-running epoch (assertable via the phase tag of the best epoch);
   the free-running phase must be able to run its full epoch budget when its
   loss is still improving, independent of how flat the teacher-forced phase
   was.

## Out of scope

- The cost of the free-running forward itself (Issue 15).
- KalmanNet (already correct — it is the pattern to copy).
- The `use_innovation_features=False` branch (no curriculum, unaffected).
