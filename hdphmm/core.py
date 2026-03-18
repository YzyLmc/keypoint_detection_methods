"""
Sticky HDP-HMM: Hierarchical Dirichlet Process Hidden Markov Model
with sticky self-transitions and AutoRegressive Gaussian observations.

Unsupervised segmentation of multiple time-series into shared behavioral
primitives using MCMC (Gibbs sampling) inference with Viterbi decoding.

Based on: Fox, Sudderth, Jordan & Willsky, "A Sticky HDP-HMM with
Application to Speaker Diarization" (Annals of Applied Statistics, 2011).

Key differences from BP-AR-HMM:
  - Shares a GLOBAL transition matrix across all demos (HDP), instead of
    per-demo feature matrices (IBP).
  - All demos use the same set of K states; the HDP governs which states
    are active and how transitions are shared.
  - Final segmentation uses Viterbi (MAP) decoding rather than the last
    Gibbs sample, giving cleaner segment boundaries.
"""

import numpy as np
from scipy import linalg

try:
    from .utils import (
        make_design_matrix,
        compute_log_likelihood,
        backward_messages,
        sample_state_sequence,
        viterbi,
        sample_inv_wishart,
        sample_matrix_normal,
        sample_dirichlet,
        _logsumexp,
    )
except ImportError:
    from utils import (
        make_design_matrix,
        compute_log_likelihood,
        backward_messages,
        sample_state_sequence,
        viterbi,
        sample_inv_wishart,
        sample_matrix_normal,
        sample_dirichlet,
        _logsumexp,
    )


class HDPHMM:
    """
    Sticky HDP-HMM with AutoRegressive Gaussian observations.

    Discovers shared behavioral states across multiple time-series
    demonstrations using Gibbs sampling, then decodes with Viterbi.

    Args:
        ar_order: Autoregressive model order (default 1).
            1 = velocity-like dynamics, 2 = acceleration-like.
        K_max: Truncation level for the number of states (default 20).
            The model will use at most K_max states; unused states die off.
            This is NOT a hard cap on discovered states -- it's a truncation
            of the infinite HDP. Set higher if you expect many distinct skills.
        gamma: Top-level DP concentration (default 5.0).
            Controls the global base measure beta. Higher = more uniform
            distribution over states. Lower = fewer states used overall.
        alpha: Transition-level DP concentration (default 5.0).
            Controls how concentrated each row of the transition matrix is
            around the global base measure beta.
        kappa: Sticky self-transition bias (default 50.0).
            Added to the diagonal of the transition matrix prior.
            Higher = longer segments, fewer state switches.
            The ratio kappa/(alpha+kappa) determines the stickiness strength.
        sF: Observation model prior scale (default 1.0).
            Scales the expected covariance of the AR observation model.
        n_iter: Number of Gibbs sampling iterations per chain (default 200).
        n_chains: Number of independent MCMC chains (default 3).
            Best chain (by log-likelihood) is selected.
        n_viterbi: Number of Viterbi re-decodings at the end (default 5).
            Final labels come from Viterbi using the last n_viterbi samples
            of parameters, selecting the one with highest log-likelihood.
        verbose: Print progress (default True).
    """

    def __init__(
        self,
        ar_order=1,
        K_max=20,
        gamma=5.0,
        alpha=5.0,
        kappa=50.0,
        sF=1.0,
        n_iter=200,
        n_chains=3,
        n_viterbi=5,
        verbose=True,
    ):
        self.ar_order = ar_order
        self.K_max = K_max
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.sF = sF
        self.n_iter = n_iter
        self.n_chains = n_chains
        self.n_viterbi = n_viterbi
        self.verbose = verbose

        # Results
        self.labels_ = None
        self.theta_ = None

    def fit(self, obs_list):
        """
        Run Sticky HDP-HMM inference on a list of observation sequences.

        Args:
            obs_list: list of N arrays, each (d, T_i) -- multi-dim time series.

        Returns:
            self
        """
        self.N = len(obs_list)
        self.d = obs_list[0].shape[0]
        self.m = self.d * self.ar_order

        self.obs_list = obs_list
        self.X_list = [make_design_matrix(obs, self.ar_order) for obs in obs_list]

        self._setup_prior()

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
        self.beta_ = best_result['beta']
        self.pi_ = best_result['pi']
        self.pi_init_ = best_result['pi_init']
        self.loglike_ = best_ll

        if self.verbose:
            n_active = len(np.unique(np.concatenate(self.labels_)))
            print(f"Best log-likelihood: {best_ll:.2f}, active states: {n_active}")

        return self

    def _setup_prior(self):
        """Set up Matrix Normal Inverse Wishart prior parameters."""
        d, m = self.d, self.m
        self.prior_M = np.zeros((d, m))
        self.prior_K = np.eye(m) * 0.1
        self.prior_nu = d + 2
        self.prior_nu_delta = np.eye(d) * self.sF

    def _run_chain(self):
        """Run a single Gibbs sampling chain."""
        N, d, m = self.N, self.d, self.m
        K = self.K_max

        # Initialize global base measure beta (stick-breaking)
        beta = np.ones(K) / K

        # Initialize global transition matrix
        pi = np.zeros((K, K))
        for j in range(K):
            pi[j] = sample_dirichlet(self.alpha * beta + self.kappa * (np.arange(K) == j))

        # Initialize starting state distribution
        pi_init = sample_dirichlet(self.alpha * beta)

        # Initialize AR parameters from prior
        theta = self._init_theta(K)

        # Initialize state sequences
        labels = self._init_labels(K)

        loglike = -np.inf

        # Store recent parameter samples for Viterbi at the end
        recent_params = []

        # First, update theta from initial labels so AR params match data
        Ustats = self._compute_sufficient_stats(labels, K)
        theta = self._sample_theta(Ustats, K)

        burn_in = self.n_iter // 3  # Don't prune during burn-in

        for it in range(self.n_iter):
            # 1. Sample state sequences z (blocked Gibbs via forward-backward)
            labels, trans_counts, init_counts = self._sample_all_states(
                theta, pi, pi_init, labels
            )

            # 2. Prune unused states (only after burn-in)
            if it >= burn_in:
                labels, theta, pi, pi_init, beta, trans_counts, init_counts, K = \
                    self._prune_states(labels, theta, pi, pi_init, beta,
                                       trans_counts, init_counts)

            # 3. Sample auxiliary variables and beta
            beta = self._sample_beta(trans_counts, beta, K)

            # 4. Sample transition matrix rows pi_j
            pi, pi_init = self._sample_transitions(trans_counts, init_counts, beta, K)

            # 5. Sample AR observation parameters theta
            Ustats = self._compute_sufficient_stats(labels, K)
            theta = self._sample_theta(Ustats, K)

            # Monitor
            if it % 10 == 0 or it == self.n_iter - 1:
                loglike = self._compute_total_loglike(theta, labels)
                if self.verbose and it % 50 == 0:
                    n_active = len(np.unique(np.concatenate(labels)))
                    print(f"  iter {it}: K_active={n_active}, loglike={loglike:.1f}")

            # Save recent parameters for Viterbi
            if it >= self.n_iter - self.n_viterbi:
                recent_params.append({
                    'theta': {k: v.copy() for k, v in theta.items()},
                    'pi': pi.copy(),
                    'pi_init': pi_init.copy(),
                    'K': K,
                })

        # Final segmentation: Viterbi decode with best recent parameters
        labels, loglike = self._viterbi_decode_best(recent_params)

        return {
            'labels': labels,
            'theta': theta,
            'beta': beta,
            'pi': pi,
            'pi_init': pi_init,
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
            block_size = max(1, T // min(K, 8))
            z = np.zeros(T, dtype=int)
            for t in range(T):
                z[t] = min(t // block_size, K - 1)
            labels.append(z)
        return labels

    def _sample_all_states(self, theta, pi, pi_init, labels):
        """
        Sample state sequences for all demos using blocked Gibbs
        (forward-backward sampling).
        """
        K = pi.shape[0]
        skills = np.arange(K)

        log_pi = np.log(np.maximum(pi, 1e-300))
        log_pi_init = np.log(np.maximum(pi_init, 1e-300))

        trans_counts = np.zeros((K, K))
        init_counts = np.zeros(K)
        new_labels = []

        for i in range(self.N):
            obs_i = self.obs_list[i]
            X_i = self.X_list[i]

            log_lik = compute_log_likelihood(obs_i, X_i, theta, skills)
            log_bwd, _ = backward_messages(log_lik, log_pi, log_pi_init)
            z = sample_state_sequence(log_lik, log_pi, log_pi_init, log_bwd)
            new_labels.append(z)

            init_counts[z[0]] += 1
            for t in range(1, len(z)):
                trans_counts[z[t - 1], z[t]] += 1

        return new_labels, trans_counts, init_counts

    def _prune_states(self, labels, theta, pi, pi_init, beta,
                      trans_counts, init_counts):
        """Remove states that are not used by any demo."""
        all_labels = np.concatenate(labels)
        used = np.unique(all_labels)
        K_old = pi.shape[0]

        if len(used) == K_old:
            return labels, theta, pi, pi_init, beta, trans_counts, init_counts, K_old

        # Keep only used states
        K_new = len(used)
        remap = {int(old): new for new, old in enumerate(used)}

        new_theta = {
            'A': theta['A'][:, :, used],
            'Sigma': theta['Sigma'][:, :, used],
            'invSigma': theta['invSigma'][:, :, used],
        }

        new_pi = pi[np.ix_(used, used)]
        # Re-normalize rows
        row_sums = new_pi.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        new_pi = new_pi / row_sums

        new_pi_init = pi_init[used]
        s = new_pi_init.sum()
        if s > 0:
            new_pi_init /= s
        else:
            new_pi_init = np.ones(K_new) / K_new

        new_beta = beta[used]
        s = new_beta.sum()
        if s > 0:
            new_beta /= s
        else:
            new_beta = np.ones(K_new) / K_new

        new_trans = trans_counts[np.ix_(used, used)]
        new_init = init_counts[used]

        new_labels = []
        for z in labels:
            new_labels.append(np.array([remap[int(v)] for v in z]))

        return (new_labels, new_theta, new_pi, new_pi_init, new_beta,
                new_trans, new_init, K_new)

    def _sample_beta(self, trans_counts, beta, K):
        """
        Sample global base measure beta using the auxiliary variable method.

        Following Teh et al. (2006): for each (j,k), sample auxiliary
        variable m_{jk} ~ number of "tables" in a Chinese Restaurant Process
        with n_{jk} customers and concentration alpha*beta_k + kappa*I(j==k).
        Then beta ~ Dir(gamma/K + sum_j m_{jk} for each k).
        """
        m_table = np.zeros((K, K))

        for j in range(K):
            for k in range(K):
                n_jk = int(trans_counts[j, k])
                if n_jk == 0:
                    continue
                conc = self.alpha * beta[k]
                if j == k:
                    conc += self.kappa
                conc = max(conc, 1e-10)
                # Sample number of tables: CRP auxiliary variable
                # m ~ sum of Bernoulli(conc / (conc + l)) for l = 0..n-1
                m = 0
                for l in range(n_jk):
                    if np.random.rand() < conc / (conc + l):
                        m += 1
                m_table[j, k] = m

        # For sticky model, split m_{jj} into "beta" and "kappa" parts
        # Each table at (j,j) was generated by beta_j w.p. alpha*beta_j/(alpha*beta_j+kappa)
        # or by kappa w.p. kappa/(alpha*beta_j+kappa)
        m_bar = m_table.copy()
        for j in range(K):
            m_jj = int(m_table[j, j])
            if m_jj > 0:
                rho = self.alpha * beta[j] / (self.alpha * beta[j] + self.kappa)
                rho = max(min(rho, 1.0 - 1e-10), 1e-10)
                # Each of the m_{jj} tables: keep with prob rho
                w = np.random.binomial(1, rho, size=m_jj)
                m_bar[j, j] = w.sum()

        # Sample beta ~ Dir(gamma/K + colsum(m_bar))
        col_sums = m_bar.sum(axis=0)
        alpha_beta = self.gamma / K + col_sums
        beta = sample_dirichlet(alpha_beta)

        return beta

    def _sample_transitions(self, trans_counts, init_counts, beta, K):
        """Sample transition matrix rows and initial distribution."""
        pi = np.zeros((K, K))
        for j in range(K):
            alpha_vec = self.alpha * beta + trans_counts[j]
            alpha_vec[j] += self.kappa
            pi[j] = sample_dirichlet(alpha_vec)

        alpha_init = self.alpha * beta + init_counts
        pi_init = sample_dirichlet(alpha_init)

        return pi, pi_init

    def _compute_sufficient_stats(self, labels, K):
        """Compute sufficient statistics for each state."""
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

            for k in range(K):
                mask = z_i == k
                if not mask.any():
                    continue
                count = mask.sum()
                Y_k = obs_i[:, mask]
                X_k = X_i[:, mask]

                Ustats['card'][k] += count
                Ustats['XX'][:, :, k] += X_k @ X_k.T
                Ustats['YX'][:, :, k] += Y_k @ X_k.T
                Ustats['YY'][:, :, k] += Y_k @ Y_k.T

        return Ustats

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

            Sxx = Ustats['XX'][:, :, k] + self.prior_K
            Syx = Ustats['YX'][:, :, k] + self.prior_M @ self.prior_K
            Syy = Ustats['YY'][:, :, k] + self.prior_M @ self.prior_K @ self.prior_M.T

            # Regularize to ensure invertibility
            Sxx += np.eye(m) * 1e-4
            try:
                Sxx_inv = linalg.inv(Sxx)
            except linalg.LinAlgError:
                try:
                    Sxx_inv = linalg.inv(Sxx + np.eye(m) * 0.1)
                except linalg.LinAlgError:
                    Sxx_inv = np.linalg.pinv(Sxx)

            M_post = Syx @ Sxx_inv
            Sygx = Syy - Syx @ Sxx_inv @ Syx.T

            Sygx = 0.5 * (Sygx + Sygx.T)
            eigvals = linalg.eigvalsh(Sygx)
            if np.min(eigvals) < 0:
                Sygx += np.eye(d) * (abs(np.min(eigvals)) + 1e-6)

            nu_post = self.prior_nu + n_k
            scale_post = Sygx + self.prior_nu_delta

            Sigma = sample_inv_wishart(nu_post, scale_post)
            if Sigma.ndim == 0:
                Sigma = np.eye(d) * float(Sigma)
            Sigma = 0.5 * (Sigma + Sigma.T)

            A = sample_matrix_normal(M_post, Sigma, Sxx_inv)

            # Ensure Sigma is PD
            eigvals_s = linalg.eigvalsh(Sigma)
            if np.min(eigvals_s) <= 0:
                Sigma += np.eye(d) * (abs(np.min(eigvals_s)) + 1e-4)

            theta['A'][:, :, k] = A
            theta['Sigma'][:, :, k] = Sigma
            try:
                theta['invSigma'][:, :, k] = linalg.inv(Sigma)
            except linalg.LinAlgError:
                theta['invSigma'][:, :, k] = np.linalg.pinv(Sigma)

        return theta

    def _viterbi_decode_best(self, recent_params):
        """
        Run Viterbi decoding using each recent parameter sample,
        return the one with highest total log-likelihood.
        """
        best_ll = -np.inf
        best_labels = None

        for params in recent_params:
            theta = params['theta']
            pi = params['pi']
            pi_init = params['pi_init']
            K = params['K']

            log_pi = np.log(np.maximum(pi, 1e-300))
            log_pi_init = np.log(np.maximum(pi_init, 1e-300))
            skills = np.arange(K)

            labels = []
            total_ll = 0.0

            for i in range(self.N):
                obs_i = self.obs_list[i]
                X_i = self.X_list[i]

                log_lik = compute_log_likelihood(obs_i, X_i, theta, skills)
                z = viterbi(log_lik, log_pi, log_pi_init)
                labels.append(z)

                # Sum log-likelihood along the decoded path
                T_i = obs_i.shape[1]
                for t in range(T_i):
                    total_ll += log_lik[z[t], t]

            if total_ll > best_ll:
                best_ll = total_ll
                best_labels = labels

        return best_labels, best_ll

    def _compute_total_loglike(self, theta, labels):
        """Compute total log-likelihood of data given current parameters."""
        total = 0.0
        K = theta['A'].shape[2]
        skills = np.arange(K)

        for i in range(self.N):
            obs_i = self.obs_list[i]
            X_i = self.X_list[i]
            z_i = labels[i]

            log_lik = compute_log_likelihood(obs_i, X_i, theta, skills)
            T_i = obs_i.shape[1]
            for t in range(T_i):
                if z_i[t] < K:
                    total += log_lik[z_i[t], t]

        return total

    def segment(self, obs_list=None):
        """
        Return the segmentation results.

        Args:
            obs_list: If None, returns results from fit(). Otherwise
                      runs Viterbi decoding on new data using fitted params.

        Returns:
            labels: list of (T_i,) arrays with state labels per timestep.
        """
        if obs_list is None:
            return self.labels_

        # Decode new data with fitted parameters
        K = self.pi_.shape[0]
        log_pi = np.log(np.maximum(self.pi_, 1e-300))
        log_pi_init = np.log(np.maximum(self.pi_init_, 1e-300))
        skills = np.arange(K)

        labels = []
        for obs in obs_list:
            X = make_design_matrix(obs, self.ar_order)
            log_lik = compute_log_likelihood(obs, X, self.theta_, skills)
            z = viterbi(log_lik, log_pi, log_pi_init)
            labels.append(z)

        return labels
