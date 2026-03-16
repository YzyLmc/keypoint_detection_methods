# BP-AR-HMM: Beta Process Autoregressive Hidden Markov Model

Python implementation of the BP-AR-HMM algorithm for **unsupervised segmentation of multiple time-series into shared behavioral primitives (skills)**.

Based on: Fox & Sudderth, *"Sharing Features among Dynamical Systems with Beta Processes"* (NIPS 2010), adapted from the [MATLAB toolbox](https://github.com/michaelchughes/NPBayesHMM) by Scott Niekum for robotics Learning from Demonstration (LfD).

## What it does

Given N multi-dimensional time-series demonstrations (e.g., robot joint trajectories), BP-AR-HMM:

1. **Discovers** a shared library of behavioral skills (unknown number, determined automatically)
2. **Segments** each demonstration into contiguous regions, each assigned to one skill
3. **Learns** the dynamics (AR model parameters) of each skill

The key property is that skills are **shared across demonstrations** -- skill 3 in demo 1 is the same skill 3 in demo 4. This is what separates BP-AR-HMM from running an HMM independently on each sequence.

## Files

| File | Description |
|------|-------------|
| `core.py` | `BPARHMM` class -- the full inference algorithm |
| `utils.py` | Math utilities: AR design matrices, likelihoods, forward-backward, samplers |
| `test.py` | Synthetic data generation and evaluation script |

## Quick start

```python
from bparhmm import BPARHMM

# obs_list: list of N arrays, each shaped (d, T_i)
# d = observation dimensionality, T_i = length of demo i
model = BPARHMM(ar_order=1, n_iter=200, n_chains=3)
model.fit(obs_list)

labels = model.segment()  # list of N integer arrays
# labels[i][t] = skill index active at time t in demo i
```

Or run the test directly:

```bash
cd bparhmm/
python3 test.py
```

This generates synthetic AR time-series with known segments, runs inference, prints accuracy, and saves a plot to `bparhmm_results.png`.

## Hyperparameters

### Most impactful (tune these first)

| Parameter | Default | What it controls | Tuning guidance |
|-----------|---------|-----------------|-----------------|
| `kappa0` | 50.0 | **Sticky self-transition bias.** Higher values penalize switching between skills, producing longer segments. | **Most important parameter.** If segments are too short/noisy, increase (100-500). If the model refuses to segment, decrease (1-10). |
| `ar_order` | 1 | **Autoregressive model order.** AR(1) uses only the previous timestep; AR(2) uses two, etc. | Use 1 for most robotics data. Increase to 2-3 if skills differ primarily in higher-order dynamics (acceleration patterns). Higher values need more data per segment. |
| `n_iter` | 200 | **MCMC iterations per chain.** More iterations = better convergence but slower. | 200 is a reasonable default. Use 500+ for complex data. Watch the log-likelihood -- if still climbing at the end, increase. |
| `n_chains` | 3 | **Independent MCMC chains.** Best chain (highest log-likelihood) is kept. | More chains = more robust to initialization. 3-5 is usually sufficient. Each chain runs the full `n_iter`. |

### Moderate impact

| Parameter | Default | What it controls | Tuning guidance |
|-----------|---------|-----------------|-----------------|
| `gamma0` | 2.0 | **IBP concentration.** Controls expected number of unique skills per demo. Higher = more skills. | Gets resampled during inference, so the initial value matters less. Set roughly to your expected number of skills. |
| `alpha0` | 1.0 | **HMM transition concentration.** Controls how uniform vs. peaked the transition distribution is. | Lower values (0.1-1) encourage sparse transitions (each skill transitions to only a few others). Higher values (5-10) allow more diffuse transitions. |
| `K_init` | 8 | **Initial number of skills.** The algorithm can grow/shrink this during inference. | Set somewhat higher than your expected true number. The birth-death RJMCMC will adjust. Too low may trap the sampler; too high just wastes some early iterations. |

### Rarely need tuning

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `sigma0` | 1.0 | Mixture component concentration (within-skill sub-states). Only relevant if you extend to multi-component mixtures. |

### Prior parameters (set in `_setup_prior()`)

These are hardcoded but can be overridden after construction:

| Attribute | Default | What it controls |
|-----------|---------|-----------------|
| `prior_M` | zeros(d, m) | Mean of AR coefficient prior. Zero = no prior bias on dynamics. |
| `prior_K` | 0.1 * I_m | Precision of AR coefficient prior. Smaller = more diffuse prior (less regularization). |
| `prior_nu` | d + 2 | Inverse-Wishart degrees of freedom. Minimum that ensures a valid prior mean exists. |
| `prior_nu_delta` | 0.1 * I_d | Inverse-Wishart scale. Controls expected noise covariance magnitude. |

To override priors:
```python
model = BPARHMM(ar_order=1)
model.d = 7  # set before calling _setup_prior
model.m = 7
model._setup_prior()
model.prior_K = np.eye(7) * 0.5  # tighter prior on AR coefficients
model.prior_nu_delta = np.eye(7) * 0.01  # expect less noise
```

## Understanding the output plot

The test script saves `bparhmm_results.png` with two columns:

- **Left column ("Ground Truth"):** Each row is one demonstration. The line plots show the time-series dimensions (up to 3 shown). Background shading indicates the **true** skill label at each timestep -- each color is a distinct skill.

- **Right column ("BP-AR-HMM"):** Same time-series, but background shading shows the **inferred** skill labels. Colors between left and right columns are **not** directly matched (the algorithm doesn't know the "true" label names), but the greedy accuracy metric in the console output accounts for the best label permutation.

**What to look for:**
- Segment boundaries in the right column should roughly align with those in the left column
- Each predicted color region should correspond consistently to one true color (even if the actual color differs)
- Over-segmentation (too many short color blocks) suggests `kappa0` is too low
- Under-segmentation (one color dominates) suggests `kappa0` is too high or `n_iter` is too low

## Algorithm overview

Each MCMC iteration performs these steps in order:

1. **Sample feature matrix F** (birth-death RJMCMC): Which skills does each demo use? Binary matrix F[i,k] = 1 if demo i uses skill k. Governed by an Indian Buffet Process (IBP) prior.

2. **Sample state sequences z** (forward-backward): For each demo, run backward message passing then forward-sample the HMM state sequence. Uses the sticky HDP-HMM transition model.

3. **Update sufficient statistics**: Aggregate XX, YX, YY matrices for each skill from the sampled state sequences.

4. **Sample transition distributions** (Dirichlet posterior): Sample pi_z (transition matrices) and pi_init (initial state) from Dirichlet posteriors with sticky bias.

5. **Sample AR parameters** (MNIW posterior): For each skill k, sample A_k (AR coefficient matrix) and Sigma_k (noise covariance) from the conjugate Matrix Normal Inverse-Wishart posterior.

6. **Resample hyperparameters**: Update IBP concentration gamma0 from its Gamma posterior.

## Implementation notes

- **Numerical stability:** Log-space arithmetic throughout. Cholesky decompositions for likelihood computation. Eigenvalue clamping for covariance matrices that drift near singular.

- **Initialization matters:** State sequences are initialized with block partitioning (evenly dividing each demo into K_init blocks). This is much better than random initialization for MCMC convergence.

- **Feature matrix sampling uses an approximation:** The original MATLAB code re-runs the full forward-backward algorithm for each feature toggle proposal. This implementation uses an approximate likelihood ratio based on how much the current state sequence uses the skill being toggled. This is faster but may be less accurate for feature discovery.

- **No mixture components (single Gaussian per skill):** The original code supports multiple mixture components per skill (the `s` variable). This implementation uses a single component per skill, which is sufficient for most robotics applications.

## Dependencies

- numpy
- scipy
- matplotlib (for test.py plotting only)