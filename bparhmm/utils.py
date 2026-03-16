"""
Utility functions for BP-AR-HMM inference.

Includes random samplers, design matrix construction, likelihood computation,
and forward-backward message passing for HMM state sequence sampling.
"""

import numpy as np
from scipy import linalg
from scipy.stats import invwishart


def make_design_matrix(obs, ar_order):
    """
    Build AR design matrix from observations.

    Args:
        obs: (d, T) observation matrix
        ar_order: AR model order r

    Returns:
        X: (d*r, T) design matrix where X[:, t] = [y_{t-1}; y_{t-2}; ...; y_{t-r}]
           For t < r, earlier lags are zero-padded.
    """
    d, T = obs.shape
    m = d * ar_order
    X = np.zeros((m, T))
    for lag in range(1, ar_order + 1):
        row_start = (lag - 1) * d
        row_end = lag * d
        if lag < T:
            X[row_start:row_end, lag:] = obs[:, :T - lag]
    return X


def compute_log_likelihood(obs, X, theta, active_skills):
    """
    Compute log-likelihood of observations under each active skill's AR model.

    Args:
        obs: (d, T) observation matrix
        X: (d*r, T) AR design matrix
        theta: dict with 'A' (d, m, K) and 'invSigma' (d, d, K) arrays
        active_skills: list of active skill indices

    Returns:
        log_lik: (len(active_skills), T) log-likelihood matrix
    """
    d, T = obs.shape
    n_active = len(active_skills)
    log_lik = np.full((n_active, T), -np.inf)

    for idx, k in enumerate(active_skills):
        A_k = theta['A'][:, :, k]          # (d, m)
        invSig_k = theta['invSigma'][:, :, k]  # (d, d)

        # Cholesky of inverse covariance for numerical stability
        try:
            L = linalg.cholesky(invSig_k, lower=True)  # invSig = L @ L^T
        except linalg.LinAlgError:
            # Fall back to eigendecomposition if not PD
            eigvals, eigvecs = linalg.eigh(invSig_k)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        log_det = np.sum(np.log(np.diag(L)))

        # Residuals: y_t - A_k @ x_t
        residuals = obs - A_k @ X  # (d, T)

        # Mahalanobis: L^T @ residual for each t
        L_residuals = L.T @ residuals  # (d, T)
        mahal = np.sum(L_residuals ** 2, axis=0)  # (T,)

        log_lik[idx, :] = -0.5 * mahal + log_det - 0.5 * d * np.log(2 * np.pi)

    return log_lik


def backward_messages(log_lik, log_pi_z, log_pi_init):
    """
    Compute backward messages for HMM.

    Args:
        log_lik: (K_active, T) log-likelihoods
        log_pi_z: (K_active, K_active) log transition matrix
        log_pi_init: (K_active,) log initial state distribution

    Returns:
        log_bwd: (K_active, T) log backward messages
        log_marginal: (K_active, T) log of bwd_msg * likelihood (partial marginals)
    """
    K, T = log_lik.shape
    log_bwd = np.zeros((K, T))
    log_marginal = np.zeros((K, T))

    # Last timestep: backward message is 1 (log = 0)
    log_marginal[:, T - 1] = log_lik[:, T - 1]

    for t in range(T - 2, -1, -1):
        # log_marginal[:, t+1] is log(beta_{t+1} * lik_{t+1})
        # log_bwd[:, t] = logsumexp over j of [log_pi_z[:, j] + log_marginal[j, t+1]]
        for i in range(K):
            log_bwd[i, t] = _logsumexp(log_pi_z[i, :] + log_marginal[:, t + 1])

        log_marginal[:, t] = log_lik[:, t] + log_bwd[:, t]

    return log_bwd, log_marginal


def sample_state_sequence(log_lik, log_pi_z, log_pi_init, log_bwd):
    """
    Forward-sample HMM state sequence given backward messages.

    Args:
        log_lik: (K_active, T)
        log_pi_z: (K_active, K_active)
        log_pi_init: (K_active,)
        log_bwd: (K_active, T)

    Returns:
        z: (T,) sampled state indices (0-indexed into active skills)
    """
    K, T = log_lik.shape
    z = np.zeros(T, dtype=int)

    # t = 0
    log_probs = log_pi_init + log_lik[:, 0] + log_bwd[:, 0]
    z[0] = _sample_log_categorical(log_probs)

    for t in range(1, T):
        log_probs = log_pi_z[z[t - 1], :] + log_lik[:, t] + log_bwd[:, t]
        z[t] = _sample_log_categorical(log_probs)

    return z


def sample_inv_wishart(df, scale):
    """
    Sample from Inverse-Wishart distribution.

    Args:
        df: degrees of freedom (scalar)
        scale: scale matrix (d, d), symmetric positive definite

    Returns:
        S: (d, d) sampled matrix
    """
    d = scale.shape[0]
    if df <= d - 1:
        df = d + 1  # ensure valid degrees of freedom
    try:
        return invwishart.rvs(df=df, scale=scale)
    except Exception:
        # Fallback: regularize scale matrix
        scale_reg = scale + np.eye(d) * 1e-6
        return invwishart.rvs(df=df, scale=scale_reg)


def sample_matrix_normal(M, Sigma, inv_K):
    """
    Sample from Matrix Normal distribution MN(M, Sigma, inv_K).

    A ~ MN(M, Sigma, V) means vec(A) ~ N(vec(M), V ⊗ Sigma)
    where V = inv(K).

    Args:
        M: (d, m) mean matrix
        Sigma: (d, d) row covariance
        inv_K: (m, m) column covariance (inverse of precision)

    Returns:
        A: (d, m) sampled matrix
    """
    d, m = M.shape
    try:
        L_sigma = linalg.cholesky(Sigma, lower=True)
    except linalg.LinAlgError:
        L_sigma = linalg.cholesky(Sigma + np.eye(d) * 1e-8, lower=True)

    try:
        L_v = linalg.cholesky(inv_K, lower=True)
    except linalg.LinAlgError:
        L_v = linalg.cholesky(inv_K + np.eye(m) * 1e-8, lower=True)

    # Sample: A = M + L_sigma @ Z @ L_v^T where Z is d x m standard normal
    Z = np.random.randn(d, m)
    A = M + L_sigma @ Z @ L_v.T
    return A


def _logsumexp(x):
    """Numerically stable logsumexp."""
    c = np.max(x)
    if np.isinf(c):
        return -np.inf
    return c + np.log(np.sum(np.exp(x - c)))


def _sample_log_categorical(log_probs):
    """Sample from categorical distribution given log-probabilities."""
    log_probs = log_probs - _logsumexp(log_probs)
    probs = np.exp(log_probs)
    probs = np.maximum(probs, 0)
    total = probs.sum()
    if total <= 0:
        return np.random.randint(len(probs))
    probs /= total
    return np.random.choice(len(probs), p=probs)


def preprocess_observations(obs_list, smooth_radius=5):
    """
    Preprocess observation sequences: smooth, center, normalize.

    Args:
        obs_list: list of (d, T_i) arrays
        smooth_radius: window radius for moving average smoothing

    Returns:
        processed: list of (d, T_i) preprocessed arrays
        stats: dict with 'mean', 'std' used for normalization
    """
    # Smooth each sequence
    smoothed = []
    for obs in obs_list:
        d, T = obs.shape
        sm = np.zeros_like(obs)
        for t in range(T):
            t_lo = max(0, t - smooth_radius)
            t_hi = min(T, t + smooth_radius + 1)
            sm[:, t] = np.mean(obs[:, t_lo:t_hi], axis=1)
        smoothed.append(sm)

    # Compute global mean
    all_obs = np.concatenate(smoothed, axis=1)
    mean = np.mean(all_obs, axis=1, keepdims=True)

    # Center
    centered = [s - mean for s in smoothed]

    # Normalize by std of first differences
    all_diffs = np.concatenate([np.diff(c, axis=1) for c in centered], axis=1)
    std = np.std(all_diffs, axis=1, keepdims=True)
    std[std < 1e-10] = 1.0

    processed = [c / std for c in centered]

    return processed, {'mean': mean, 'std': std}
