"""
BP-AR-HMM: Beta Process Autoregressive Hidden Markov Model.

Nonparametric Bayesian model for unsupervised segmentation of multiple
time-series into shared behavioral primitives/skills.

Based on: Fox & Sudderth, "Sharing Features among Dynamical Systems
with Beta Processes" (NIPS 2010).
"""

import numpy as np
from scipy import linalg

try:
    from .utils import (
        make_design_matrix,
        compute_log_likelihood,
        backward_messages,
        sample_state_sequence,
        sample_inv_wishart,
        sample_matrix_normal,
        _logsumexp,
        _sample_log_categorical,
    )
except ImportError:
    from utils import (
        make_design_matrix,
        compute_log_likelihood,
        backward_messages,
        sample_state_sequence,
        sample_inv_wishart,
        sample_matrix_normal,
        _logsumexp,
        _sample_log_categorical,
    )


class BPARHMM:
    """
    Beta Process Autoregressive Hidden Markov Model.

    Discovers shared behavioral primitives across multiple time-series
    demonstrations using MCMC inference.

    Args:
        ar_order: Autoregressive model order (default 1).
        K_init: Initial number of skills (default 8).
        alpha0: HMM transition concentration parameter (default 1.0).
        kappa0: Sticky self-transition bias (default 50.0).
        gamma0: IBP concentration parameter (default 2.0).
        sigma0: Mixture component concentration (default 1.0).
        n_iter: Number of MCMC iterations (default 200).
        n_chains: Number of independent MCMC chains (default 3).
        verbose: Print progress (default True).
    """

    def __init__(
        self,
        ar_order=1,
        K_init=8,
        alpha0=1.0,
        kappa0=50.0,
        gamma0=2.0,
        sigma0=1.0,
        n_iter=200,
        n_chains=3,
        verbose=True,
    ):
        self.ar_order = ar_order
        self.K_init = K_init
        self.alpha0 = alpha0
        self.kappa0 = kappa0
        self.gamma0 = gamma0
        self.sigma0 = sigma0
        self.n_iter = n_iter
        self.n_chains = n_chains
        self.verbose = verbose

        # Results
        self.labels_ = None
        self.theta_ = None
        self.F_ = None

    def fit(self, obs_list):
        """
        Run BP-AR-HMM inference on a list of observation sequences.

        Args:
            obs_list: list of N arrays, each (d, T_i) -- multi-dim time series.

        Returns:
            self
        """
        self.N = len(obs_list)
        self.d = obs_list[0].shape[0]
        self.m = self.d * self.ar_order

        # Build design matrices
        self.obs_list = obs_list
        self.X_list = [make_design_matrix(obs, self.ar_order) for obs in obs_list]

        # Set up MNIW prior
        self._setup_prior()

        # Run multiple chains, keep best
        best_ll = -np.inf
        best_result = None

        for chain in range(self.n_chains):
            if self.verbose:
                print(f"Chain {chain + 1}/{self.n_chains}")
            result = self._run_chain()
            if result['loglike'] > best_ll:
                best_ll = result['loglike']
                best_result = result

        self.labels_ = best_result['labels']
        self.theta_ = best_result['theta']
        self.F_ = best_result['F']
        self.loglike_ = best_ll

        if self.verbose:
            print(f"Best log-likelihood: {best_ll:.2f}")

        return self

    def _setup_prior(self):
        """Set up Matrix Normal Inverse Wishart prior parameters."""
        d, m = self.d, self.m
        # Prior mean of AR coefficients (zero = no dynamics prior)
        self.prior_M = np.zeros((d, m))
        # Prior precision on AR coefficients
        self.prior_K = np.eye(m) * 0.1
        # IW degrees of freedom
        self.prior_nu = d + 2
        # IW scale matrix
        self.prior_nu_delta = np.eye(d) * 0.1

    def _run_chain(self):
        """Run a single MCMC chain."""
        N, d, m = self.N, self.d, self.m
        K = self.K_init

        # Initialize feature matrix: all demos use all initial skills
        F = np.ones((N, K), dtype=int)

        # Initialize theta (AR parameters) from prior
        theta = self._init_theta(K)

        # Initialize state sequences with block partitioning
        labels = self._init_labels(K)

        # Initialize transition distributions
        dist = self._init_distributions(F)

        # Hyperparameters that get resampled
        gamma0 = self.gamma0

        loglike = -np.inf

        for it in range(self.n_iter):
            # 1. Sample feature matrix F (birth-death RJMCMC)
            F, theta, dist, labels = self._sample_features(
                F, gamma0, theta, dist, labels
            )
            K = F.shape[1]

            # 2. Sample state sequences z
            labels, state_counts = self._sample_all_state_seqs(F, theta, dist)

            # 3. Update sufficient statistics
            Ustats = self._compute_sufficient_stats(labels, K)

            # 4. Sample transition distributions
            dist = self._sample_distributions(state_counts, F)

            # 5. Sample AR parameters theta
            theta = self._sample_theta(Ustats, K)

            # 6. Resample IBP concentration
            gamma0 = self._sample_gamma(F, gamma0)

            # Compute log-likelihood for monitoring
            if it % 10 == 0 or it == self.n_iter - 1:
                loglike = self._compute_total_loglike(F, theta, labels)
                if self.verbose and it % 50 == 0:
                    print(f"  iter {it}: K={K}, loglike={loglike:.1f}")

        return {
            'labels': labels,
            'theta': theta,
            'F': F,
            'loglike': loglike,
        }

    def _init_theta(self, K):
        """Initialize AR parameters by sampling from prior."""
        d, m = self.d, self.m
        theta = {
            'A': np.zeros((d, m, K)),
            'Sigma': np.zeros((d, d, K)),
            'invSigma': np.zeros((d, d, K)),
        }
        for k in range(K):
            Sigma = sample_inv_wishart(self.prior_nu, self.prior_nu_delta)
            if Sigma.ndim == 0:
                Sigma = np.eye(d) * float(Sigma)
            theta['Sigma'][:, :, k] = Sigma
            theta['invSigma'][:, :, k] = linalg.inv(Sigma)
            A = sample_matrix_normal(self.prior_M, Sigma, linalg.inv(self.prior_K))
            theta['A'][:, :, k] = A
        return theta

    def _init_labels(self, K):
        """Initialize state sequences with block partitioning."""
        labels = []
        for i in range(self.N):
            T = self.obs_list[i].shape[1]
            block_size = max(1, T // K)
            z = np.zeros(T, dtype=int)
            for t in range(T):
                z[t] = min(t // block_size, K - 1)
            labels.append(z)
        return labels

    def _init_distributions(self, F):
        """Initialize transition distributions."""
        K = F.shape[1]
        dist = []
        for i in range(self.N):
            active = np.where(F[i] > 0)[0]
            Ka = len(active)
            if Ka == 0:
                Ka = 1
                active = np.array([0])
            # Sticky transitions
            pi_z = np.ones((Ka, Ka)) * self.alpha0 / Ka
            for j in range(Ka):
                pi_z[j, j] += self.kappa0
            # Normalize rows
            pi_z = pi_z / pi_z.sum(axis=1, keepdims=True)
            pi_init = np.ones(Ka) / Ka
            dist.append({
                'pi_z': pi_z,
                'pi_init': pi_init,
                'active_skills': active.copy(),
            })
        return dist

    def _sample_features(self, F, gamma0, theta, dist, labels):
        """
        Sample feature matrix F using birth-death RJMCMC.

        For each demo i, propose toggling each feature, and propose
        birth/death of unique features.
        """
        N, K = F.shape

        for i in range(N):
            # Count how many other demos use each feature
            other_counts = F.sum(axis=0) - F[i]

            for k in range(F.shape[1]):
                if other_counts[k] > 0:
                    # Shared feature: MH toggle
                    # Must have at least one active feature after toggle
                    if F[i, k] == 1 and F[i].sum() <= 1:
                        continue

                    # IBP prior ratio: p(f_ik=1)/p(f_ik=0) = m_{-i,k} / (N - m_{-i,k})
                    # where m_{-i,k} = number of OTHER demos using feature k
                    m_ik = other_counts[k]
                    if F[i, k] == 1:
                        # Proposing to turn off: ratio = p(0)/p(1)
                        log_prior_ratio = np.log(N - m_ik) - np.log(m_ik)
                    else:
                        # Proposing to turn on: ratio = p(1)/p(0)
                        log_prior_ratio = np.log(m_ik) - np.log(N - m_ik)

                    # Likelihood ratio: check how much this skill is used in labels
                    count_k = np.sum(labels[i] == k)
                    if F[i, k] == 1 and count_k > 0:
                        # Turning off an actively used skill is costly
                        log_lik_ratio = -count_k * 2.0
                    elif F[i, k] == 0:
                        # Turning on: small bonus only if many demos use it
                        log_lik_ratio = 0.0
                    else:
                        log_lik_ratio = 0.0

                    log_accept = log_prior_ratio + log_lik_ratio
                    if np.log(np.random.rand()) < log_accept:
                        old_val = F[i, k]
                        F[i, k] = 1 - old_val
                        # Reassign labels that used this skill if we turned it off
                        if old_val == 1:
                            active = np.where(F[i] > 0)[0]
                            if len(active) > 0:
                                mask = labels[i] == k
                                if mask.any():
                                    labels[i][mask] = np.random.choice(
                                        active, size=mask.sum()
                                    )

            # Birth-death moves for unique features (used only by demo i)
            unique_mask = (F.sum(axis=0) == F[i]) & (F[i] == 1)
            n_unique = unique_mask.sum()

            # Death move: try to kill a unique feature
            if n_unique > 0 and np.random.rand() < 0.5:
                unique_indices = np.where(unique_mask)[0]
                kill_idx = np.random.choice(unique_indices)
                if F[i].sum() > 1:
                    active = np.where(F[i] > 0)[0]
                    active = active[active != kill_idx]
                    mask = labels[i] == kill_idx
                    if mask.any() and len(active) > 0:
                        labels[i][mask] = np.random.choice(active, size=mask.sum())
                    F[i, kill_idx] = 0

            # Birth move: propose a new unique feature for this demo
            elif np.random.rand() < gamma0 / N:
                new_col = np.zeros((N, 1), dtype=int)
                new_col[i, 0] = 1
                F = np.hstack([F, new_col])
                theta = self._extend_theta(theta, 1)

        # Prune: remove features unused in ANY label sequence
        # This prevents phantom features from accumulating
        K = F.shape[1]
        used_in_labels = np.zeros(K, dtype=bool)
        for i in range(N):
            for k in np.unique(labels[i]):
                if k < K:
                    used_in_labels[k] = True
        # Also keep features that are active in F even if not in labels yet
        used_in_F = F.sum(axis=0) > 0
        # A feature survives if it's in F AND used in labels, or if it's shared
        shared = F.sum(axis=0) >= 2
        keep_mask = (used_in_F & used_in_labels) | shared
        # Always keep at least one feature
        if not keep_mask.any():
            keep_mask[0] = True

        if not np.all(keep_mask):
            keep_indices = np.where(keep_mask)[0]
            F = F[:, keep_indices]
            theta['A'] = theta['A'][:, :, keep_indices]
            theta['Sigma'] = theta['Sigma'][:, :, keep_indices]
            theta['invSigma'] = theta['invSigma'][:, :, keep_indices]
            # Remap labels
            remap = {int(old): new for new, old in enumerate(keep_indices)}
            for i in range(N):
                new_labels = np.zeros_like(labels[i])
                for t in range(len(labels[i])):
                    old_label = int(labels[i][t])
                    if old_label in remap:
                        new_labels[t] = remap[old_label]
                    else:
                        active = np.where(F[i] > 0)[0]
                        new_labels[t] = active[0] if len(active) > 0 else 0
                labels[i] = new_labels

        # Ensure every demo has at least one active feature
        for i in range(N):
            if F[i].sum() == 0:
                F[i, 0] = 1

        dist = self._init_distributions(F)
        return F, theta, dist, labels

    def _extend_theta(self, theta, n_new):
        """Extend theta arrays by n_new skills sampled from prior."""
        d, m = self.d, self.m
        for _ in range(n_new):
            Sigma = sample_inv_wishart(self.prior_nu, self.prior_nu_delta)
            if Sigma.ndim == 0:
                Sigma = np.eye(d) * float(Sigma)
            invSigma = linalg.inv(Sigma)
            A = sample_matrix_normal(self.prior_M, Sigma, linalg.inv(self.prior_K))
            theta['A'] = np.concatenate(
                [theta['A'], A[:, :, np.newaxis]], axis=2
            )
            theta['Sigma'] = np.concatenate(
                [theta['Sigma'], Sigma[:, :, np.newaxis]], axis=2
            )
            theta['invSigma'] = np.concatenate(
                [theta['invSigma'], invSigma[:, :, np.newaxis]], axis=2
            )
        return theta

    def _sample_all_state_seqs(self, F, theta, dist):
        """Sample state sequences for all demonstrations."""
        N = self.N
        K = F.shape[1]
        labels = []
        # Aggregate state counts
        state_counts = {
            'N': np.zeros((K, K)),     # transition counts
            'N0': np.zeros(K),         # initial state counts
        }

        for i in range(N):
            active = np.where(F[i] > 0)[0]
            Ka = len(active)
            if Ka == 0:
                labels.append(np.zeros(self.obs_list[i].shape[1], dtype=int))
                continue

            obs_i = self.obs_list[i]
            X_i = self.X_list[i]
            T_i = obs_i.shape[1]

            # Compute log-likelihoods for active skills
            log_lik = compute_log_likelihood(obs_i, X_i, theta, active)

            # Build transition matrices for active skills
            pi_z = dist[i]['pi_z']
            pi_init = dist[i]['pi_init']

            log_pi_z = np.log(np.maximum(pi_z, 1e-300))
            log_pi_init = np.log(np.maximum(pi_init, 1e-300))

            # Backward messages
            log_bwd, _ = backward_messages(log_lik, log_pi_z, log_pi_init)

            # Forward sampling
            z_local = sample_state_sequence(log_lik, log_pi_z, log_pi_init, log_bwd)

            # Map local indices back to global skill indices
            z_global = np.array([active[z_local[t]] for t in range(T_i)])
            labels.append(z_global)

            # Accumulate state counts
            state_counts['N0'][z_global[0]] += 1
            for t in range(1, T_i):
                state_counts['N'][z_global[t - 1], z_global[t]] += 1

        return labels, state_counts

    def _compute_sufficient_stats(self, labels, K):
        """Compute sufficient statistics for each skill."""
        d, m = self.d, self.m
        Ustats = {
            'card': np.zeros(K),
            'XX': np.zeros((m, m, K)),
            'YX': np.zeros((d, m, K)),
            'YY': np.zeros((d, d, K)),
        }

        for i in range(self.N):
            obs_i = self.obs_list[i]
            X_i = self.X_list[i]
            z_i = labels[i]
            T_i = obs_i.shape[1]

            for k in range(K):
                mask = z_i == k
                if not mask.any():
                    continue
                count = mask.sum()
                Y_k = obs_i[:, mask]  # (d, count)
                X_k = X_i[:, mask]    # (m, count)

                Ustats['card'][k] += count
                Ustats['XX'][:, :, k] += X_k @ X_k.T
                Ustats['YX'][:, :, k] += Y_k @ X_k.T
                Ustats['YY'][:, :, k] += Y_k @ Y_k.T

        return Ustats

    def _sample_distributions(self, state_counts, F):
        """Sample transition distributions given state counts."""
        K = F.shape[1]
        dist = []

        for i in range(self.N):
            active = np.where(F[i] > 0)[0]
            Ka = len(active)
            if Ka == 0:
                Ka = 1
                active = np.array([0])

            # Sample pi_z (transition matrix) with sticky prior
            pi_z = np.zeros((Ka, Ka))
            for a_idx in range(Ka):
                k = active[a_idx]
                counts = np.zeros(Ka)
                for b_idx in range(Ka):
                    j = active[b_idx]
                    counts[b_idx] = state_counts['N'][k, j]
                # Dirichlet posterior
                alpha_vec = np.ones(Ka) * self.alpha0 / Ka
                alpha_vec[a_idx] += self.kappa0  # sticky bias
                alpha_vec += counts
                pi_z[a_idx, :] = np.random.dirichlet(np.maximum(alpha_vec, 1e-10))

            # Sample pi_init
            init_counts = np.zeros(Ka)
            for a_idx in range(Ka):
                init_counts[a_idx] = state_counts['N0'][active[a_idx]]
            alpha_init = np.ones(Ka) * self.alpha0 / Ka + init_counts
            pi_init = np.random.dirichlet(np.maximum(alpha_init, 1e-10))

            dist.append({
                'pi_z': pi_z,
                'pi_init': pi_init,
                'active_skills': active.copy(),
            })

        return dist

    def _sample_theta(self, Ustats, K):
        """Sample AR parameters from MNIW posterior."""
        d, m = self.d, self.m
        theta = {
            'A': np.zeros((d, m, K)),
            'Sigma': np.zeros((d, d, K)),
            'invSigma': np.zeros((d, d, K)),
        }

        for k in range(K):
            n_k = Ustats['card'][k]

            # Posterior sufficient statistics
            Sxx = Ustats['XX'][:, :, k] + self.prior_K
            Syx = Ustats['YX'][:, :, k] + self.prior_M @ self.prior_K
            Syy = Ustats['YY'][:, :, k] + self.prior_M @ self.prior_K @ self.prior_M.T

            # Posterior parameters
            try:
                Sxx_inv = linalg.inv(Sxx)
            except linalg.LinAlgError:
                Sxx_inv = linalg.inv(Sxx + np.eye(m) * 1e-6)

            M_post = Syx @ Sxx_inv
            Sygx = Syy - Syx @ Sxx_inv @ Syx.T

            # Ensure positive definiteness
            Sygx = 0.5 * (Sygx + Sygx.T)
            eigvals = linalg.eigvalsh(Sygx)
            if np.min(eigvals) < 0:
                Sygx += np.eye(d) * (abs(np.min(eigvals)) + 1e-6)

            nu_post = self.prior_nu + n_k
            scale_post = Sygx + self.prior_nu_delta

            # Sample Sigma ~ IW(nu_post, scale_post)
            Sigma = sample_inv_wishart(nu_post, scale_post)
            if Sigma.ndim == 0:
                Sigma = np.eye(d) * float(Sigma)
            Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrize

            # Sample A | Sigma ~ MN(M_post, Sigma, Sxx_inv)
            A = sample_matrix_normal(M_post, Sigma, Sxx_inv)

            theta['A'][:, :, k] = A
            theta['Sigma'][:, :, k] = Sigma
            try:
                theta['invSigma'][:, :, k] = linalg.inv(Sigma)
            except linalg.LinAlgError:
                theta['invSigma'][:, :, k] = linalg.inv(
                    Sigma + np.eye(d) * 1e-6
                )

        return theta

    def _sample_gamma(self, F, gamma0):
        """Resample IBP concentration parameter gamma0."""
        N, K = F.shape
        # Number of active features
        K_plus = (F.sum(axis=0) > 0).sum()
        # Gamma prior: Gamma(a, b)
        a_gamma, b_gamma = 1.0, 1.0
        # Posterior: Gamma(a + K_plus, b + H_N) where H_N = sum(1/i for i=1..N)
        H_N = np.sum(1.0 / np.arange(1, N + 1))
        gamma_new = np.random.gamma(a_gamma + K_plus, 1.0 / (b_gamma + H_N))
        return max(gamma_new, 0.01)

    def _compute_total_loglike(self, F, theta, labels):
        """Compute total log-likelihood of data given current parameters."""
        total = 0.0
        for i in range(self.N):
            obs_i = self.obs_list[i]
            X_i = self.X_list[i]
            z_i = labels[i]
            T_i = obs_i.shape[1]

            for t in range(T_i):
                k = z_i[t]
                if k >= theta['A'].shape[2]:
                    continue
                A_k = theta['A'][:, :, k]
                invSig_k = theta['invSigma'][:, :, k]
                residual = obs_i[:, t] - A_k @ X_i[:, t]
                try:
                    sign, logdet = np.linalg.slogdet(invSig_k)
                    if sign <= 0:
                        logdet = 0.0
                    total += -0.5 * residual @ invSig_k @ residual + 0.5 * logdet
                except Exception:
                    total += -0.5 * np.sum(residual ** 2)

        return total

    def segment(self, obs_list=None):
        """
        Return the segmentation results.

        Args:
            obs_list: If None, returns results from fit(). Otherwise re-labels
                      using the fitted model parameters.

        Returns:
            labels: list of (T_i,) arrays with skill labels for each timestep.
        """
        if obs_list is None:
            return self.labels_
        raise NotImplementedError("Segmenting new data with a fitted model is not yet supported.")
