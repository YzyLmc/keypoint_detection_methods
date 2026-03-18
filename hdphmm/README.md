# Sticky HDP-HMM: Hierarchical Dirichlet Process Hidden Markov Model

Python implementation of the **Sticky HDP-HMM** algorithm for unsupervised segmentation of multiple time-series into shared behavioral states.

Based on: Fox, Sudderth, Jordan & Willsky, *"A Sticky HDP-HMM with Application to Speaker Diarization"* (Annals of Applied Statistics, 2011). Adapted from the [bnpy](https://github.com/bnpy/bnpy)-based change point detection pipeline, reimplemented from scratch with Gibbs sampling and Viterbi decoding.

## What it does

Given N multi-dimensional time-series demonstrations (e.g., robot end-effector trajectories), Sticky HDP-HMM:

1. **Discovers** a shared set of behavioral states (number determined automatically up to a truncation level)
2. **Segments** each demonstration into contiguous regions, each assigned to one state
3. **Learns** the dynamics (AR model parameters) of each state
4. **Decodes** the final segmentation using the Viterbi algorithm for clean MAP segment boundaries

States are **shared across all demonstrations** via the Hierarchical Dirichlet Process -- state 2 in demo 1 is the same dynamical regime as state 2 in demo 5.

## Comparison with BP-AR-HMM

| Aspect | BP-AR-HMM (`bparhmm/`) | Sticky HDP-HMM (`hdphmm/`) |
|--------|------------------------|----------------------------|
| State sharing | **IBP** (Indian Buffet Process): per-demo binary feature matrix F[i,k] controls which skills each demo can use | **HDP** (Hierarchical Dirichlet Process): all demos share a global transition matrix; states are shared implicitly |
| Transition model | Per-demo transition matrices, sticky via kappa0 on diagonal | Global transition matrix, sticky via kappa on diagonal |
| Number of states | Grows/shrinks via birth-death RJMCMC moves on feature matrix | Fixed truncation level K_max; unused states are pruned after burn-in |
| Final segmentation | Last Gibbs sample | **Viterbi MAP decoding** (cleaner boundaries) |
| Inference | Gibbs sampling | Gibbs sampling + Viterbi |
| Original source | MATLAB (NPBayesHMM toolbox) | Python (bnpy library, variational Bayes) |

Both use the same AR Gaussian observation model and the same MNIW (Matrix Normal Inverse Wishart) posterior updates.

## Files

| File | Description |
|------|-------------|
| `core.py` | `HDPHMM` class -- the full inference algorithm |
| `utils.py` | Math utilities: AR design matrices, likelihoods, forward-backward, Viterbi, samplers |
| `test.py` | Synthetic data generation and evaluation script |
| `run_robomimic.py` | Apply HDP-HMM to robomimic robot demonstration datasets |

## Quick start

```python
from hdphmm import HDPHMM

# obs_list: list of N arrays, each shaped (d, T_i)
# d = observation dimensionality, T_i = length of demo i
model = HDPHMM(ar_order=1, K_max=20, n_iter=200, n_chains=3)
model.fit(obs_list)

labels = model.segment()  # list of N integer arrays
# labels[i][t] = state index active at time t in demo i
```

Or run the test directly:

```bash
cd hdphmm/
python3 test.py
```

This generates synthetic AR time-series with known segments, runs inference, prints accuracy, and saves a comparison plot to `test_hdphmm_results.png`.

### Applying to robomimic data

```bash
cd hdphmm/
python3 run_robomimic.py \
    --dataset /tmp/core_datasets/coffee/demo_src_coffee_task_D0/demo.hdf5 \
    --output_dir hdphmm/results_robomimic
```

Add `--per_demo` to also save individual per-demo segmentation plots.

## Hyperparameters

### Most impactful (tune these first)

| Parameter | Default | What it controls | Tuning guidance |
|-----------|---------|-----------------|-----------------|
| `kappa` | 50.0 | **Sticky self-transition bias.** Added to the diagonal of the transition matrix prior. Higher = longer segments, fewer state switches. The ratio `kappa/(alpha+kappa)` determines the stickiness strength. | **Most important parameter.** If segments are too short/noisy, increase (100-500). If the model refuses to segment, decrease (1-10). |
| `ar_order` | 1 | **Autoregressive model order.** AR(1) uses only the previous timestep; AR(2) uses two, etc. | Use 1 for most data. Increase to 2 if states differ primarily in higher-order dynamics (acceleration patterns). Higher values need more data per segment. |
| `K_max` | 20 | **Truncation level.** Maximum number of states the model can use. Unused states are pruned after burn-in. | Set higher than your expected number of states. Not a hard cap -- the model will naturally use fewer. Too low may prevent discovery; too high just wastes early compute. |
| `n_iter` | 200 | **Gibbs sampling iterations per chain.** More iterations = better convergence but slower. | 200 is a reasonable default. Use 300+ for complex data. Watch the log-likelihood -- if still climbing at the end, increase. |
| `n_chains` | 3 | **Independent MCMC chains.** Best chain (highest log-likelihood) is kept. | More chains = more robust to initialization. 3-5 is usually sufficient. Each chain runs the full `n_iter`. |

### Moderate impact

| Parameter | Default | What it controls | Tuning guidance |
|-----------|---------|-----------------|-----------------|
| `gamma` | 5.0 | **Top-level DP concentration.** Controls the global base measure beta, which determines how state usage is distributed overall. Higher = more uniform state usage. Lower = fewer states used. | If you're getting too many states, decrease. If too few, increase. |
| `alpha` | 5.0 | **Transition-level DP concentration.** Controls how concentrated each row of the transition matrix is around the global base measure beta. | Lower values (0.5-2) encourage each state to transition to only a few next-states. Higher values (5-10) allow more diffuse transitions. |
| `sF` | 1.0 | **Observation model prior scale.** Scales the expected covariance of the AR observation model via the IW scale matrix. | Higher = more diffuse prior on observation noise. If your data has small/large variance, adjust accordingly. |
| `n_viterbi` | 5 | **Viterbi averaging window.** Number of recent parameter samples used for final Viterbi decoding. The decoding with the highest log-likelihood is selected. | 3-10 is reasonable. Higher values are more robust but slower. |

### Prior parameters (set in `_setup_prior()`)

These are hardcoded but can be overridden after construction:

| Attribute | Default | What it controls |
|-----------|---------|-----------------|
| `prior_M` | zeros(d, m) | Mean of AR coefficient prior. Zero = no prior bias on dynamics. |
| `prior_K` | 0.1 * I_m | Precision of AR coefficient prior. Smaller = more diffuse prior (less regularization). |
| `prior_nu` | d + 2 | Inverse-Wishart degrees of freedom. Minimum that ensures a valid prior mean exists. |
| `prior_nu_delta` | sF * I_d | Inverse-Wishart scale. Controls expected noise covariance magnitude. |

## Data preprocessing

When applying to robomimic data (`run_robomimic.py`), the input is an 8D absolute pose per timestep:

```
[x, y, z]         from obs/robot0_eef_pos    (end-effector position)
[qx, qy, qz, qw]  from obs/robot0_eef_quat   (end-effector orientation)
[gripper]          from actions[:, -1]         (gripper openness)
```

**Normalization** (first-difference normalization):
1. Compute first differences (s_t - s_{t-1}) across all demos
2. Compute the standard deviation of these deltas per dimension
3. Scale each dimension by its delta std so that delta variance = 1.0
4. The gripper dimension uses the **median** of the pose dimensions' delta stds instead of its own (because gripper deltas are mostly zero, giving a misleadingly small std)
5. Center each dimension at zero

This normalization ensures all dimensions contribute equally to the AR model, regardless of their natural scale.

## Output format

The segmentation is stored as a dictionary compatible with the BP-AR-HMM pipeline:

```python
segmentation_dict = {
    'demo_0': {
        (0, 45): 2,      # timesteps 0-45: state 2
        (46, 120): 0,     # timesteps 46-120: state 0
        (121, 180): 1,    # timesteps 121-180: state 1
    },
    'demo_1': { ... },
    ...
}
```

Saved as both `segmentation.pkl` (for Python) and `segmentation.json` (human-readable).

## Algorithm overview

### Gibbs sampling (each iteration)

1. **Sample state sequences z** (forward-backward): For each demo, compute log-likelihoods under all states, run backward message passing, then forward-sample the state sequence. All demos share the same transition matrix and AR parameters.

2. **Prune unused states** (after burn-in): Remove states that no demo is currently using. Remap labels and compact arrays.

3. **Sample beta** (auxiliary variable method): Sample the global base measure beta using the Chinese Restaurant Process auxiliary variable technique (Teh et al. 2006), with correction for the sticky term.

4. **Sample transition matrix pi** (Dirichlet posterior): For each state j, sample row pi_j from Dir(alpha * beta + n_j + kappa * delta_j), where n_j counts observed transitions from state j.

5. **Sample AR parameters** (MNIW posterior): For each state k, aggregate sufficient statistics (XX, YX, YY) from all timesteps assigned to state k across all demos, then sample (A_k, Sigma_k) from the conjugate Matrix Normal Inverse Wishart posterior.

### Viterbi decoding (final step)

After Gibbs sampling converges, the final segmentation is obtained by Viterbi MAP decoding rather than using the last Gibbs sample. This produces cleaner segment boundaries. The last `n_viterbi` parameter samples are each used for Viterbi decoding, and the one with the highest total log-likelihood is selected.

### Key algorithmic details

- **Burn-in**: State pruning is disabled during the first 1/3 of iterations to prevent premature collapse. During burn-in, the model can explore many states freely.

- **Sticky transitions**: The self-transition bias kappa is added to the diagonal of the Dirichlet prior for each transition row, encouraging the model to stay in the same state. This is critical for producing temporally coherent segments rather than rapid switching.

- **Beta sampling with sticky correction**: When sampling the auxiliary variables m_{jk} for the global base measure, the self-transition tables m_{jj} are split into "beta-generated" and "kappa-generated" portions using a Bernoulli filter. Only the beta-generated tables contribute to the beta update. This prevents the sticky term from inflating the global base measure.

## Understanding the output plot

The test script saves `test_hdphmm_results.png` with two columns:

- **Left ("True"):** Background shading shows ground-truth state labels
- **Right ("Predicted"):** Background shading shows inferred state labels

Colors are not directly matched between columns (the algorithm discovers label indices independently), but the accuracy metric uses best-permutation matching.

**Diagnostic signs:**
- Over-segmentation (too many short segments) → increase `kappa`
- Under-segmentation (one state dominates) → decrease `kappa` or increase `n_iter`
- Too many/few states discovered → adjust `gamma` or `K_max`

## Dependencies

- numpy
- scipy
- matplotlib (for plotting only)
- h5py (for `run_robomimic.py` only)
