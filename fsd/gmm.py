"""GPU-accelerated Gaussian Mixture Model using PyTorch.

Faithful reimplementation of sklearn's GaussianMixture EM algorithm.
Produces identical results (within float64 tolerance) when given the same
initialization. Supports "full" and "tied" covariance types.

Usage:
    from fsd.gmm import TorchGMM, load_gmm

    # Inference from pre-trained weights
    gmm = load_gmm("weights/gmm.pt", device="cuda")
    log_lik = gmm.score_samples(X)

    # Training a new GMM
    gmm = TorchGMM(n_components=5, covariance_type="tied", device="cuda")
    gmm.fit(X)
"""

import math
import numpy as np
import torch

from scipy.special import logsumexp as scipy_logsumexp


class TorchGMM:
    """GPU-accelerated Gaussian Mixture Model matching sklearn's algorithm.

    Parameters match sklearn.mixture.GaussianMixture for drop-in replacement.
    """

    def __init__(
        self,
        n_components=5,
        covariance_type="full",
        tol=1e-6,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        random_state=2026,
        verbose=0,
        device="cuda",
    ):
        if covariance_type not in ("full", "tied"):
            raise ValueError(f"covariance_type must be 'full' or 'tied', got '{covariance_type}'")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device(device)

        # Fitted attributes (set by fit())
        self.means_ = None
        self.weights_ = None
        self.covariances_ = None
        self.precisions_cholesky_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -math.inf

        # Precomputed inference cache (set by _update_inference_cache)
        self._log_det_ = None
        self._log_weights_ = None
        self._log_const_ = None
        self._means_prec_ = None

    def to(self, device):
        """Move all fitted parameters to a new device. Returns self."""
        device = torch.device(device)
        self.device = device
        for attr in ("means_", "weights_", "covariances_", "precisions_cholesky_"):
            v = getattr(self, attr, None)
            if v is not None:
                setattr(self, attr, v.to(device))
        self._update_inference_cache()
        return self

    def _to_tensor(self, X):
        """Convert input to float64 tensor on self.device."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.to(dtype=torch.float64, device=self.device)

    def _update_inference_cache(self):
        """Precompute derived quantities for fast score_samples."""
        if self.precisions_cholesky_ is None:
            return

        pc = self.precisions_cholesky_
        if self.covariance_type == "full":
            self._log_det_ = pc.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            K = self.n_components
            self._means_prec_ = torch.stack([self.means_[k] @ pc[k] for k in range(K)])
        else:  # tied
            self._log_det_ = pc.diagonal().log().sum()
            self._means_prec_ = self.means_ @ pc

        self._log_weights_ = self.weights_.log()

        D = self.means_.shape[1]
        log_const = -0.5 * D * math.log(2 * math.pi)
        self._log_const_ = log_const + self._log_det_

    def _initialize(self, X, random_state):
        """KMeans initialization matching sklearn's default init_params='kmeans'."""
        from sklearn.cluster import KMeans

        N, D = X.shape
        K = self.n_components

        X_cpu = X.cpu().numpy()
        km = KMeans(n_clusters=K, n_init=1, random_state=random_state)
        labels = km.fit_predict(X_cpu)

        resp = torch.zeros(N, K, dtype=torch.float64, device=self.device)
        resp[torch.arange(N), torch.from_numpy(labels).to(self.device)] = 1.0

        nk, means, covariances = self._estimate_gaussian_parameters(X, resp)

        self.weights_ = nk / N
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = self._compute_precision_cholesky(covariances)
        self._update_inference_cache()

    def _estimate_gaussian_parameters(self, X, resp):
        """Compute nk, means, covariances from responsibilities."""
        eps = torch.finfo(torch.float64).eps
        nk = resp.sum(dim=0) + 10 * eps
        means = (resp.T @ X) / nk.unsqueeze(1)

        if self.covariance_type == "full":
            covariances = self._estimate_covariances_full(X, resp, nk, means)
        else:
            covariances = self._estimate_covariances_tied(X, resp, nk, means)

        return nk, means, covariances

    def _estimate_covariances_full(self, X, resp, nk, means):
        """Compute per-component covariances. Returns (K, D, D)."""
        K = self.n_components
        D = X.shape[1]
        covariances = torch.empty(K, D, D, dtype=torch.float64, device=self.device)

        for k in range(K):
            diff = X - means[k]
            covariances[k] = (resp[:, k].unsqueeze(0) * diff.T) @ diff / nk[k]
            covariances[k].diagonal().add_(self.reg_covar)

        return covariances

    def _estimate_covariances_tied(self, X, resp, nk, means):
        """Compute tied (shared) covariance. Returns (D, D)."""
        K = self.n_components
        D = X.shape[1]
        covariance = torch.zeros(D, D, dtype=torch.float64, device=self.device)

        for k in range(K):
            diff = X - means[k]
            covariance += (resp[:, k].unsqueeze(0) * diff.T) @ diff

        covariance /= nk.sum()
        covariance.diagonal().add_(self.reg_covar)
        return covariance

    def _compute_precision_cholesky(self, covariances):
        """Compute upper-triangular precision Cholesky from covariances."""
        if self.covariance_type == "full":
            K, D, _ = covariances.shape
            L = torch.linalg.cholesky(covariances)
            I_K = torch.eye(D, dtype=torch.float64, device=self.device).expand(K, -1, -1)
            L_inv = torch.linalg.solve_triangular(L, I_K, upper=False)
            return L_inv.mT
        else:  # tied
            D = covariances.shape[0]
            L = torch.linalg.cholesky(covariances)
            I_D = torch.eye(D, dtype=torch.float64, device=self.device)
            L_inv = torch.linalg.solve_triangular(L, I_D, upper=False)
            return L_inv.T

    def _estimate_log_prob(self, X):
        """Compute per-component log Gaussian probabilities. Returns (N, K)."""
        N, D = X.shape
        K = self.n_components
        log_det = self._log_det_ if self._log_det_ is not None else self._compute_log_det_raw()

        log_prob = torch.empty(N, K, dtype=torch.float64, device=self.device)

        if self.covariance_type == "full":
            for k in range(K):
                prec_chol = self.precisions_cholesky_[k]
                y = X @ prec_chol - self.means_[k] @ prec_chol
                log_prob[:, k] = (y * y).sum(dim=1)
        else:  # tied
            prec_chol = self.precisions_cholesky_
            X_transformed = X @ prec_chol
            for k in range(K):
                y = X_transformed - self.means_[k] @ prec_chol
                log_prob[:, k] = (y * y).sum(dim=1)

        log_const = -0.5 * D * math.log(2 * math.pi)
        if self.covariance_type == "full":
            return log_const + log_det.unsqueeze(0) - 0.5 * log_prob
        else:
            return log_const + log_det - 0.5 * log_prob

    def _compute_log_det_raw(self):
        """Fallback: compute log det without cache."""
        pc = self.precisions_cholesky_
        if self.covariance_type == "full":
            return pc.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        else:
            return pc.diagonal().log().sum()

    def _estimate_weighted_log_prob(self, X):
        """Log prob + log weights. Returns (N, K)."""
        return self._estimate_log_prob(X) + self.weights_.log()

    def _e_step(self, X):
        """E-step: compute responsibilities.

        Returns:
            mean_log_likelihood: scalar (the lower bound)
            log_resp: (N, K) log responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        log_resp = weighted_log_prob - log_prob_norm.unsqueeze(1)
        return log_prob_norm.mean().item(), log_resp

    def _m_step(self, X, log_resp):
        """M-step: update parameters from responsibilities."""
        resp = log_resp.exp()
        nk, means, covariances = self._estimate_gaussian_parameters(X, resp)

        self.weights_ = nk / nk.sum()
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = self._compute_precision_cholesky(covariances)
        self._update_inference_cache()

    def _get_parameters(self):
        """Snapshot current parameters."""
        return {
            "weights": self.weights_.clone(),
            "means": self.means_.clone(),
            "covariances": self.covariances_.clone(),
            "precisions_cholesky": self.precisions_cholesky_.clone(),
        }

    def _set_parameters(self, params):
        """Restore parameters from snapshot."""
        self.weights_ = params["weights"]
        self.means_ = params["means"]
        self.covariances_ = params["covariances"]
        self.precisions_cholesky_ = params["precisions_cholesky"]
        self._update_inference_cache()

    def fit(self, X):
        """Fit GMM via EM algorithm.

        Requires scikit-learn for KMeans initialization.
        Install with: pip install fsd-detector[fit]

        Args:
            X: (N, D) tensor or numpy array.

        Returns:
            self
        """
        X = self._to_tensor(X)
        N, D = X.shape

        if N < self.n_components:
            raise ValueError(
                f"Expected n_samples >= n_components, got {N} samples and {self.n_components} components"
            )

        rng = np.random.RandomState(self.random_state)
        max_lower_bound = -math.inf
        best_params = None
        best_n_iter = 0

        for init_idx in range(self.n_init):
            if self.verbose >= 1:
                print(f"Initialization {init_idx + 1}/{self.n_init}")

            self._initialize(X, random_state=rng)

            lower_bound = -math.inf
            converged = False

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                lower_bound, log_resp = self._e_step(X)
                self._m_step(X, log_resp)

                change = lower_bound - prev_lower_bound

                if self.verbose >= 2:
                    print(f"  Iteration {n_iter:4d}  LL={lower_bound:.6f}  change={change:+.2e}")

                if abs(change) < self.tol:
                    converged = True
                    if self.verbose >= 1:
                        print(f"  Converged at iteration {n_iter} (change={change:+.2e})")
                    break

            if self.verbose >= 1 and not converged:
                print(f"  Did not converge after {self.max_iter} iterations")

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter
                self.converged_ = converged

        if best_params is not None:
            self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Final E-step (matches sklearn)
        _, log_resp = self._e_step(X)

        return self

    def score_samples(self, X):
        """Per-sample log-likelihood.

        Uses numpy on CPU for minimal overhead, PyTorch on CUDA for GPU speed.

        Args:
            X: (N, D) tensor or numpy array.

        Returns:
            (N,) tensor of log p(x_i) on self.device.
        """
        if self.device.type == "cpu":
            return self._score_samples_numpy(X)
        else:
            X = self._to_tensor(X)
            with torch.no_grad():
                return torch.logsumexp(self._estimate_weighted_log_prob(X), dim=1)

    def _score_samples_numpy(self, X):
        """Fast CPU inference using numpy (avoids PyTorch dispatch overhead)."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        K = self.n_components

        prec_chol = self.precisions_cholesky_.numpy()
        means_prec = self._means_prec_.numpy()
        log_weights = self._log_weights_.numpy()

        if self.covariance_type == "full":
            log_const = self._log_const_.numpy()
        else:
            log_const = float(self._log_const_)

        weighted_log_prob = np.empty((N, K), dtype=np.float64)

        if self.covariance_type == "full":
            for k in range(K):
                y = X @ prec_chol[k] - means_prec[k]
                mahal_sq = np.sum(y * y, axis=1)
                weighted_log_prob[:, k] = log_const[k] - 0.5 * mahal_sq + log_weights[k]
        else:  # tied
            X_transformed = X @ prec_chol
            for k in range(K):
                y = X_transformed - means_prec[k]
                mahal_sq = np.sum(y * y, axis=1)
                weighted_log_prob[:, k] = log_const - 0.5 * mahal_sq + log_weights[k]

        result = scipy_logsumexp(weighted_log_prob, axis=1)
        return torch.from_numpy(result)

    def score(self, X):
        """Mean log-likelihood (scalar)."""
        return self.score_samples(X).mean().item()

    def predict(self, X):
        """Predict component labels.

        Args:
            X: (N, D) tensor or numpy array.

        Returns:
            (N,) tensor of component indices.
        """
        X = self._to_tensor(X)
        with torch.no_grad():
            _, log_resp = self._e_step(X)
        return log_resp.argmax(dim=1)

    def predict_proba(self, X):
        """Predict posterior probabilities.

        Args:
            X: (N, D) tensor or numpy array.

        Returns:
            (N, K) tensor of responsibilities.
        """
        X = self._to_tensor(X)
        with torch.no_grad():
            _, log_resp = self._e_step(X)
        return log_resp.exp()

    def bic(self, X):
        """Bayesian Information Criterion."""
        X = self._to_tensor(X)
        N = X.shape[0]
        return -2 * self.score(X) * N + self._n_parameters(X.shape[1]) * math.log(N)

    def aic(self, X):
        """Akaike Information Criterion."""
        X = self._to_tensor(X)
        N = X.shape[0]
        return -2 * self.score(X) * N + 2 * self._n_parameters(X.shape[1])

    def _n_parameters(self, D):
        """Number of free parameters."""
        K = self.n_components
        mean_params = K * D
        weight_params = K - 1
        if self.covariance_type == "full":
            cov_params = K * D * (D + 1) // 2
        else:  # tied
            cov_params = D * (D + 1) // 2
        return mean_params + weight_params + cov_params


def load_gmm(path, device="cpu"):
    """Load a pre-trained GMM from a .pt weights file.

    Args:
        path: Path to the .pt file containing GMM parameters.
        device: Device to load onto.

    Returns:
        TorchGMM instance ready for inference.
    """
    data = torch.load(path, map_location="cpu", weights_only=True)

    gmm = TorchGMM(
        n_components=int(data["n_components"]),
        covariance_type=data["covariance_type"],
        device="cpu",
    )
    gmm.means_ = data["means_"]
    gmm.weights_ = data["weights_"]
    gmm.covariances_ = data["covariances_"]
    gmm.precisions_cholesky_ = data["precisions_cholesky_"]
    gmm._update_inference_cache()

    if device != "cpu":
        gmm.to(device)

    return gmm
