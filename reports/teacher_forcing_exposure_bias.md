# Report: Do the neural filters train the way they are deployed?

**Question (user):** The Transformer, Mamba, Neural-ODE and PINN don't seem to
train sequentially / on their own inputs. Doesn't that break causality or make
them worse in real deployment (error-accumulation) than in training (teacher
forcing)?

**Short answer:** The concern is **correct for 2 of the 5** neural filters —
**Transformer** and **Mamba** — which are trained *teacher-forced only* and
deployed *free-running*, a real exposure-bias / error-accumulation mismatch. The
other three — **PINN**, **Neural-ODE**, and **KalmanNet** — already train
free-running (or with a free-running curriculum phase), so they do **not** have
this gap. Causality itself is not broken in any of them.

Filed as **[Issue 13](../issue/13_exposure_bias_transformer_mamba.md)**.

---

## 1. Two things the question conflates (both worth separating)

1. **Causality** — can `x_hat_t` see the future `y_{>t}`? No, in all five.
   The Transformer uses a causal mask ([`transformer.py:84-85`](../estimators/neural/transformer.py#L84-L85)),
   Mamba's scan and the recurrent models are causal by construction. Training is
   parallel *over time* but still causal. So "parallel training breaks causality"
   is **not** happening.

2. **Teacher forcing vs. free-running** — at each step, is the *input* built from
   the **ground-truth** previous state (teacher forcing) or the model's **own**
   previous estimate (free-running)? This is where the real issue lives, and it
   differs per estimator.

The user's instinct — "they don't train on their own input, so they'll do worse
under real error accumulation" — is exactly the definition of **exposure bias**,
and it applies precisely to the teacher-forced-only models.

## 2. Per-estimator verdict

| Estimator | Training input built from | Inference input built from | Mismatch? |
|---|---|---|---|
| **Transformer** | **ground-truth** prev state → `f`,`h` | model's own `x_hat_{t-1}` | **YES** (exposure bias) |
| **Mamba** | **ground-truth** prev state → `f`,`h` | model's own `x_hat_{t-1}` | **YES** (exposure bias) |
| PINN | its **own** `x_hat_{t-1}` (free-running) | its own `x_hat_{t-1}` | no |
| Neural-ODE | its **own** `x_post` (free-running) | its own `x_post` | no |
| KalmanNet | Phase 1 GT, then **Phase 2 free-running** | own `x_hat_{t-1}` | mitigated by curriculum |

Evidence:
- Transformer teacher-forced input: [`transformer.py:194-200`](../estimators/neural/transformer.py#L194-L200);
  free-running inference: [`transformer.py:222-235`](../estimators/neural/transformer.py#L222-L235) (`x_prev = x_hat`).
- Mamba: same split — [`mamba.py:323-328`](../estimators/neural/mamba.py#L323-L328) (train, GT) vs.
  [`mamba.py:345-364`](../estimators/neural/mamba.py#L345-L364) (infer, `x_prev = x_hat`).
- PINN free-running training loop: [`pinn.py:139-146`](../estimators/neural/pinn.py#L139-L146) (`x = x_pred + dx`, fed back).
- Neural-ODE free-running training loop: [`neural_ode.py:209-217`](../estimators/neural/neural_ode.py#L209-L217) (`x = x_post`, fed back).
- KalmanNet two-phase curriculum: [`kalmannet.py:136-150`](../estimators/neural/kalmannet.py#L136-L150),
  free-running Phase 2 at [`kalmannet.py:317-335`](../estimators/neural/kalmannet.py#L317-L335).

## 3. Why the Transformer/Mamba design chose teacher forcing

Both parallelize over the time axis only when the per-step input is **independent
of the network weights** — the Transformer via one masked attention pass, Mamba
via the associative selective scan. Building the input from ground-truth prev
state makes it weight-independent, so the whole `[B, T, *]` sequence trains in one
parallel pass (and can even be cached — Issue 9). Feeding the model's own output
back would serialize training into a T-step loop, forfeiting the very speedup
those architectures exist for. So the mismatch is a deliberate speed/fidelity
trade — but it is currently **undocumented and unmeasured**, which is the real
problem.

## 4. Why it actually bites here (not the benign NLP case)

Teacher forcing is standard and usually harmless for language decoders, because at
inference you feed back **observed-token-shaped** values that are in-distribution
once the model is decent. Here the fed-back quantity is a **latent state estimate**
that has *no ground truth at deployment*, and `f`/`h` are nonlinear — so a small
state error maps `x_pred`/`innovation` **off** the distribution the network trained
on. Errors then compound step to step. This is the recursive-estimation regime
where exposure bias genuinely degrades accuracy, worst on chaotic (Lorenz) and
weakly-observable levels.

Caveat: with `use_innovation_features=False` the Transformer/Mamba consume raw
`[y, dt]` (no fed-back state), so that configuration has **no** exposure bias — but
also no process-model conditioning. The default (`True`) is the affected mode.

## 5. Impact on the benchmark's fairness

The suite reports deployment RMSE under strictly-sequential CPU inference. A
teacher-forced-only model's number is **optimistic** versus an equivalent
honestly-trained model, and the models penalized are exactly the two strong
sequence models (Transformer, Mamba) — while KalmanNet/PINN/Neural-ODE are trained
free-running. So the current comparison is not apples-to-apples on the axis that
matters most (robustness to self-generated error).

## 6. Recommended fix (see Issue 13 for detail)

- **Preferred:** add a **free-running fine-tune phase** to Transformer & Mamba,
  mirroring KalmanNet's Phase-2 curriculum — keep the fast teacher-forced pass as a
  warm-start, then anneal into training on the model's own estimate (input
  construction identical to `_estimate_sequential_cpu`).
- **Alternative:** scheduled sampling (anneal ground-truth → own-estimate mixing).
- **Minimum bar if the sequential cost is rejected:** document both models as
  teacher-forced-only and add a benchmark diagnostic reporting the
  teacher-forced-vs-free-running RMSE gap per level, so the optimism is visible.

Do **not** modify PINN / Neural-ODE / KalmanNet training — they already train the
way they are deployed.
