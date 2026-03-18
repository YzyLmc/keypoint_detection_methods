"""
Utility functions for Sticky HDP-HMM inference.

Includes random samplers, design matrix construction, likelihood computation,
forward-backward message passing, and Viterbi decoding.
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


def compute_log_likelihood(obs, X, theta, skills):
    """
    Compute log-likelihood of observations under each skill's AR model.

    Args:
        obs: (d, T) observation matrix
        X: (d*r, T) AR design matrix
        theta: dict with 'A' (d, m, K) and 'invSigma' (d, d, K) arrays
        skills: array of skill indices to evaluate

    Returns:
        log_lik: (len(skills), T) log-likelihood matrix
    """
    d, T = obs.shape
    n_skills = len(skills)
    log_lik = np.full((n_skills, T), -np.inf)

    for idx, k in enumerate(skills):
        A_k = theta['A'][:, :, k]
        invSig_k = theta['invSigma'][:, :, k]

        # Check for degenerate inverse covariance
        if np.any(np.isnan(invSig_k)) or np.any(np.isinf(invSig_k)):
            log_lik[idx, :] = -1e6
            continue

        try:
            L = _safe_cholesky(invSig_k, d)
        except Exception:
            log_lik[idx, :] = -1e6
            continue

        diag_L = np.diag(L)
        diag_L = np.maximum(np.abs(diag_L), 1e-10)
        log_det = np.sum(np.log(diag_L))

        residuals = obs - A_k @ X
        L_residuals = L.T @ residuals
        mahal = np.sum(L_residuals ** 2, axis=0)
        ll = -0.5 * mahal + log_det - 0.5 * d * np.log(2 * np.pi)
        # Clamp to avoid extreme values
        log_lik[idx, :] = np.clip(ll, -1e6, 1e6)

    return log_lik


def backward_messages(log_lik, log_pi_z, log_pi_init):
    """
    Compute backward messages for HMM.

    Args:
        log_lik: (K, T) log-likelihoods
        log_pi_z: (K, K) log transition matrix
        log_pi_init: (K,) log initial state distribution

    Returns:
        log_bwd: (K, T) log backward messages
        log_marginal: (K, T) log(bwd * lik)
    """
    K, T = log_lik.shape
    log_bwd = np.zeros((K, T))
    log_marginal = np.zeros((K, T))

    log_marginal[:, T - 1] = log_lik[:, T - 1]

    for t in range(T - 2, -1, -1):
        for i in range(K):
            log_bwd[i, t] = _logsumexp(log_pi_z[i, :] + log_marginal[:, t + 1])
        log_marginal[:, t] = log_lik[:, t] + log_bwd[:, t]

    return log_bwd, log_marginal


def sample_state_sequence(log_lik, log_pi_z, log_pi_init, log_bwd):
    """
    Forward-sample HMM state sequence given backward messages.

    Args:
        log_lik: (K, T) log-likelihoods
        log_pi_z: (K, K) log transition matrix
        log_pi_init: (K,) log initial state distribution
        log_bwd: (K, T) log backward messages

    Returns:
        z: (T,) sampled state indices
    """
    K, T = log_lik.shape
    z = np.zeros(T, dtype=int)

    log_probs = log_pi_init + log_lik[:, 0] + log_bwd[:, 0]
    z[0] = _sample_log_categorical(log_probs)

    for t in range(1, T):
        log_probs = log_pi_z[z[t - 1], :] + log_lik[:, t] + log_bwd[:, t]
        z[t] = _sample_log_categorical(log_probs)

    return z


def viterbi(log_lik, log_pi_z, log_pi_init):
    """
    Viterbi algorithm for MAP state sequence.

    Args:
        log_lik: (K, T) log-likelihoods
        log_pi_z: (K, K) log transition matrix
        log_pi_init: (K,) log initial state distribution

    Returns:
        z: (T,) MAP state sequence
    """
    K, T = log_lik.shape
    delta = np.full((K, T), -np.inf)
    psi = np.zeros((K, T), dtype=int)

    delta[:, 0] = log_pi_init + log_lik[:, 0]

    for t in range(1, T):
        for j in range(K):
            scores = delta[:, t - 1] + log_pi_z[:, j]
            psi[j, t] = np.argmax(scores)
            delta[j, t] = scores[psi[j, t]] + log_lik[j, t]

    # Backtrack
    z = np.zeros(T, dtype=int)
    z[T - 1] = np.argmax(delta[:, T - 1])
    for t in range(T - 2, -1, -1):
        z[t] = psi[z[t + 1], t + 1]

    return z


def sample_inv_wishart(df, scale):
    """
    Sample from Inverse-Wishart distribution.

    Args:
        df: degrees of freedom
        scale: (d, d) scale matrix

    Returns:
        S: (d, d) sampled matrix
    """
    d = scale.shape[0]
    if df <= d - 1:
        df = d + 1
    # Ensure scale is symmetric positive definite
    scale = 0.5 * (scale + scale.T)
    eigvals = linalg.eigvalsh(scale)
    if np.min(eigvals) <= 0:
        scale += np.eye(d) * (abs(np.min(eigvals)) + 1e-4)
    try:
        return invwishart.rvs(df=df, scale=scale)
    except Exception:
        scale_reg = scale + np.eye(d) * 1e-3
        try:
            return invwishart.rvs(df=df, scale=scale_reg)
        except Exception:
            return np.eye(d) * np.mean(np.diag(scale)) / df


def _safe_cholesky(A, n):
    """Cholesky decomposition with robust fallback for near-singular matrices."""
    A = 0.5 * (A + A.T)
    for reg in [0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
        try:
            return linalg.cholesky(A + np.eye(n) * reg, lower=True)
        except linalg.LinAlgError:
            continue
    # Last resort: eigendecomposition
    eigvals, eigvecs = linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-4)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def sample_matrix_normal(M, Sigma, inv_K):
    """
    Sample from Matrix Normal distribution MN(M, Sigma, inv_K).

    vec(A) ~ N(vec(M), inv_K kron Sigma)

    Args:
        M: (d, m) mean matrix
        Sigma: (d, d) row covariance
        inv_K: (m, m) column covariance

    Returns:
        A: (d, m) sampled matrix
    """
    d, m = M.shape
    L_sigma = _safe_cholesky(Sigma, d)
    L_v = _safe_cholesky(inv_K, m)

    Z = np.random.randn(d, m)
    A = M + L_sigma @ Z @ L_v.T
    return A


def sample_dirichlet(alpha):
    """
    Sample from Dirichlet distribution, handling small alpha values.

    Args:
        alpha: (K,) concentration parameters

    Returns:
        p: (K,) sampled probability vector
    """
    alpha = np.maximum(alpha, 1e-10)
    p = np.random.dirichlet(alpha)
    p = np.maximum(p, 1e-300)
    p /= p.sum()
    return p


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
