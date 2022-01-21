import numpy as np

# MATH and STATS:
import math
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from scipy.stats._multivariate import _PSD
import scipy
from scipy.special import softmax, logsumexp

# for initialization of cluster's centers:
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
# for initial imputation of X
from sklearn.impute import KNNImputer


def compute_bic(n, k, p, LL):
    n_parameters = k*p + k-1 + k*p*(p+1)*0.5
    bic = -2*LL + n_parameters * np.log(n)

    return bic


def HDDA_regularization(S_full, regularizator):
    EPS = np.finfo(np.float).eps
    # HIGH DIMENSIONAL CLUSTERING https://doi.org/10.1016/j.csda.2007.02.009
    Lk, Qk = scipy.linalg.eigh(S_full, lower=False)
    Lk[Lk < EPS] = EPS
    idx = Lk.argsort()[::-1]
    Lk, Qk = Lk[idx], Qk[:, idx]
    # Scree test
    dL, dk = np.absolute(np.diff(Lk)), 1
    dL /= dL.max()
    while np.any(dL[dk:] > regularizator):
        dk += 1
    # bk replace small eigval
    bk = 1 / (S_full.shape[1] - dk) * (S_full.trace() - np.sum(Lk[0:dk]))
    bk = np.max([EPS, bk])

    Lk[dk:] = bk
    Lk = Lk * np.eye(Lk.shape[0])
    S_reg = Qk @ Lk @ Qk.T

    return S_reg / np.trace(S_reg)


def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """
    Determine which eigenvalues are "small" given the spectrum.

    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.
    This is designed to be compatible with scipy.linalg.pinvh.

    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.

    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.

    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps


class FEM():
    '''Implements the F-EM algorithm

    Parameters
    ----------
    K : int
        The number of mixture components.
    max_iter: int
        maximum number of iterations of the algorithm.
    rand_initialization: bool
        True if random initialization
        False if K-Means initialization.
     max_iter_fp: integer>0
        maximum number of fixed-point iterations


    Attributes
    ----------
    alpha_ : array-like, shape (n,)
        The weight of each mixture components.
    mu_ : array-like, shape (n, p)
        The mean of each mixture component.
    Sigma_ : array-like, shape (p, p)
        The covariance of each mixture component.
    tau_ : array-like, shape (n, K)
        The collection of tau values.
    labels_ : array-like, shape (n,)
        The clustering labels of the data points.
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of step used to reach the convergence.
    '''

    def __init__(self, K, max_iter=200,
                 rand_initialization=False,
                 version=1, max_iter_fp=20, thres=None,
                 HDDA=False, reg=1e-6):
        self.K = K
        self.converged_ = False
        self.version = version
        self.rand_initialization = rand_initialization
        self.max_iter = max_iter
        self.max_iter_fp = max_iter_fp
        self.thres = thres
        self.alpha_ = None
        self.mu_ = None
        self.Sigma_ = None
        self.n_iter_ = None
        self.labels_ = None
        self.X_hat = None
        self.X_hat_all = None
        self.S_hat_all = None
        self.M = None
        self.observed_rows = None
        self.non_observed_rows = None
        self.HDDA = HDDA
        self.reg = reg
        self.bic = None

    def _initialize(self, X):
        '''Initialize all the parameters of the model:
        theta = (alpha, mu, sigma, tau)
        Either randomly or with kmeans centers.

        Parameters
        ----------
        X: array-like, shape (n, p)

        '''
        n, p = X.shape
        self.M = ~np.isnan(X)  # true where not nan
        self.observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
        self.non_observed_rows = np.where(~np.isnan(sum(X.T)) == False)[0]

        X_init = KNNImputer().fit_transform(X)
        self.X_hat = X_init

        if self.rand_initialization:
            self.alpha_ = np.random.rand(3)
            self.alpha_ /= np.sum(self.alpha_)
            self.mu_ = (np.amax(X_init, axis=0) - np.amin(X, axis=0)) * np.random.random_sample((self.K, p)) + np.amin(X_init, axis=0)
            self.Sigma_ = np.zeros((self.K, p, p))
            for k in range(self.K):
                self.Sigma_[k] = np.eye(p)
        else:
            one_point_clusters = False
            kmeans = KMeans(n_clusters=self.K, max_iter=200).fit(X_init)

            for k in range(self.K):
                nk = np.count_nonzero(kmeans.labels_ == k)
                if nk <= 2 and n > 10:
                    one_point_clusters = True

            ite_filter = 0
            n_filter = n

            if one_point_clusters:
                print("One point cluster")
                tree = cKDTree(X_init)  # tree of nearest neighbors
                KNN = 4
                dd, index = tree.query(X_init, k=[
                    KNN])  # query for all points in data the Kth NN, returns distances and indexes
                dd = np.reshape(dd, (n,))
                alpha_quantile = 0.95

                while one_point_clusters and alpha_quantile > 0.5:
                    ite_filter += 1
                    alpha_quantile -= (0.1) * (ite_filter - 1)
                    one_point_clusters = False
                    X_without_extremes = X_init[dd < np.quantile(dd, alpha_quantile), :]
                    n_filter = X_without_extremes.shape[0]
                    kmeans = KMeans(n_clusters=self.K, max_iter=200).fit(X_without_extremes)
                    for k in range(self.K):
                        nk = np.count_nonzero(kmeans.labels_ == k)
                        if nk <= 2:
                            one_point_clusters = True
            self.alpha_ = np.zeros((self.K,))
            self.mu_ = np.zeros((self.K, p))
            self.Sigma_ = np.zeros((self.K, p, p))
            for k in range(self.K):
                nk = np.count_nonzero(kmeans.labels_ == k)
                self.alpha_[k] = float(nk) / float(n_filter)
                self.mu_[k] = kmeans.cluster_centers_[k]
                # self.Sigma_[k] = np.eye(p)  # cov result in nan sometimes

                # CHANGE
                # adding Sigma = cov
                labels = kmeans.labels_
                idx_cluster = np.where(labels == k)[0]
                observed_rows_cluster = idx_cluster[np.where(np.isnan(sum(X[idx_cluster, :].T)) == False)[0]]

                self.Sigma_[k] = np.cov(X_init[observed_rows_cluster,].T)
                if np.isnan( self.Sigma_[k]).any():
                    self.Sigma_[k] = np.diag(np.nanvar(X_init[idx_cluster, :], axis=0))
                    if np.isnan(self.Sigma_[k]).any():
                        self.Sigma_[k] = np.eye(p)

                # We check that Sigma can be inverted
                s, u = scipy.linalg.eigh(self.Sigma_[k], lower=True, check_finite=True)
                small_e = _eigvalsh_to_eps(s, None, None)
                if np.min(s) < -small_e:
                    print("min", np.min(s))
                    print("eps", small_e)
                    print('the input matrix must be positive semidefinite')
                    self.Sigma_[k] += (- np.min(s) + 50 * small_e) * np.eye(self.Sigma_[k].shape)
                    s, u = scipy.linalg.eigh(self.Sigma_[k], lower=True, check_finite=True)
                    small_e = _eigvalsh_to_eps(s, None, None)
                    print("MIN new")
                    print(np.min(s))

    def _e_step(self, X):
        ''' E-step of the algorithm
        Computes the conditional probability of the model

        Parameters
        ----------
        X: array-like, shape (n, p)
            data

        Returns
        ----------
        cond_prob_matrix: array-like, shape (n, K)
             (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)
        '''
        n, p = X.shape

        K = len(self.alpha_)
        log_cond_prob_matrix = np.zeros((n, K))

        X_hat_all = np.zeros([self.X_hat.shape[0], self.X_hat.shape[1], self.K])
        for k in range(self.K):
            X_hat_all[:, :, k] = self.X_hat
        S_hat_all = np.zeros([self.X_hat.shape[1], self.X_hat.shape[1], self.X_hat.shape[0], self.K])

        for k in range(K):
            diff = X[self.observed_rows, :] - self.mu_[k]
            # sq_maha = (np.dot(diff, np.linalg.inv(self.Sigma_[k])) * diff).sum(1)
            # logdet = np.linalg.slogdet(self.Sigma_[k])[1]
            psd = _PSD(self.Sigma_[k], allow_singular=True)
            sq_maha = np.sum(np.square(np.dot(diff, psd.U)), axis=-1) + 10 ** (-18)
            logdet = psd.log_pdet
            logdensity = np.log(self.alpha_[k]) - 0.5 * logdet - 0.5 * p * np.log(sq_maha)
            log_cond_prob_matrix[self.observed_rows, k] = logdensity

            # Rows with missing data
            for i in self.non_observed_rows:
                X_o = self.X_hat[i, self.M[i, :]]
                sigma_oo = self.Sigma_[k][np.ix_(self.M[i, :], self.M[i, :])]
                mu_o = self.mu_[k][self.M[i, :]]
                p_obs = sum(self.M[i, :])  # observed_features

                diff = X_o - mu_o
                psd = _PSD(sigma_oo, allow_singular=True)
                sq_maha = np.sum(np.square(np.dot(diff, psd.U)), axis=-1) + 10 ** (-18)
                logdet = psd.log_pdet
                logdensity = np.log(self.alpha_[k]) - 0.5 * logdet - 0.5 * p_obs * np.log(sq_maha)
                log_cond_prob_matrix[i, k] = logdensity

                # filling missing values and compute expectations
                # missing parts
                sigma_mo = self.Sigma_[k][np.ix_(~self.M[i, :], self.M[i, :])]
                sigma_mm = self.Sigma_[k][np.ix_(~self.M[i, :], ~self.M[i, :])]
                sigma_om = self.Sigma_[k][np.ix_(self.M[i, :], ~self.M[i, :])]
                mu_m = self.mu_[k][~self.M[i, :]]

                S_hat_temp = S_hat_all[:, :, i, k]
                sigma_oo_inv = psd.pinv

                X_hat_all[i, ~self.M[i, :], k] = mu_m + sigma_mo @ sigma_oo_inv @ (X_o - mu_o)  # check
                S_hat_temp_small = sigma_mm - sigma_mo @ sigma_oo_inv @ sigma_om
                S_hat_temp[np.ix_(~self.M[i, :], ~self.M[i, :])] = S_hat_temp_small
                S_hat_all[:, :, i, k] = S_hat_temp

        self.X_hat_all = X_hat_all
        self.S_hat_all = S_hat_all

        cond_prob_matrix = softmax(log_cond_prob_matrix, axis=1)
        LL = logsumexp(log_cond_prob_matrix, axis=1).sum()
        self.bic = compute_bic(n=X.shape[0], k=K, p=X.shape[1], LL=LL)

        # filling X_hat by expectations
        for i in self.non_observed_rows:
            self.X_hat[i, ~self.M[i, :]] = 0
            for k in range(self.K):
                self.X_hat[i, ~self.M[i, :]] += cond_prob_matrix[i, k] * X_hat_all[i, ~self.M[i, :], k]

        return cond_prob_matrix

    def _m_step(self, X, cond_prob):
        ''' M-step of the algorithm
        Updates all the parameters with the new conditional probabilities

        Parameters
        ----------
        X: array-like, shape (n, p)
            data
        cond_prob_matrix: array-like, shape (n, K)
             (cond_prob_matrix)_ik = P(Z_i=k|X_i=x_i)

        Returns
        ----------
        alpha_new: array-like, shape (n,)
            The new weights of each mixture components.
        mu_new: array-like, shape (n, p)
            The new mean of each mixture component.
        Sigma_new: array-like, shape (p, p)
            The new covariance of each mixture component.
        tau_new: array-like, shape (n, K)
            The collection of tau values.
        '''

        n, p = X.shape

        alpha_new = np.zeros((self.K,))
        mu_new = np.zeros((self.K, p))
        Sigma_new = np.zeros((self.K, p, p))
        tau_new = np.ones((n, self.K))

        for k in range(self.K):
            # UPDATE alpha:
            alpha_new[k] = np.mean(cond_prob[:, k])
            # Fixed-point equation for Sigma and mu:
            # UPDATE mu
            # UPDATE Sigma
            mu_fixed_point = self.mu_[k].copy()
            Sigma_fixed_point = self.Sigma_[k].copy()

            gamma_k = cond_prob[:, k]  # probas component k
            S_hat_k = self.S_hat_all[:, :, :, k] # expectations S^mm
            X_hat_k = self.X_hat_all[:, :, k]  # expectations X^mm

            convergence_fp = False
            ite_fp = 1
            while not convergence_fp and ite_fp < self.max_iter_fp:
                # Computing weights for each samples ~ 1/tau_i
                diff = X_hat_k - mu_fixed_point
                psd = _PSD(Sigma_fixed_point, allow_singular=True)
                sq_maha = np.sum(np.square(np.dot(diff, psd.U)), axis=-1)
                # remove extreme values...
                sq_maha = np.where(sq_maha < 10 ** (-18), 10 ** (-18),
                                   np.where(sq_maha > 10 ** (18), 10 ** (18), sq_maha))
                weights = 1/sq_maha

                gamma_k[gamma_k < 0] = 0
                weights[weights < 0] = 0

                if np.sum(gamma_k) == 0:
                    gamma_k[gamma_k == 0] = 1
                if np.sum(weights) == 0:
                    weights[weights == 0] = 1

                # new mu
                Xp = (X_hat_k.T * (gamma_k * weights)).T
                mu_fixed_point_new = np.sum(Xp, axis=0) / np.sum(gamma_k * weights)

                # new Sigma
                Sigma_fixed_point_inv = psd.pinv
                S_hat_fixed_point = S_hat_k.copy()

                for i in self.non_observed_rows:
                    trace_temp = ((Sigma_fixed_point_inv * S_hat_fixed_point[:, :, i].T).sum())
                    if trace_temp != 0:
                        S_hat_fixed_point[:, :, i] = S_hat_fixed_point[:, :, i] / trace_temp

                if len(self.non_observed_rows) != 0:
                    S_left = np.sum(gamma_k * S_hat_fixed_point, axis=2) / np.sum(gamma_k)
                else:
                    S_left = 0

                S_right = np.dot(cond_prob[:, k] * weights * diff.T, diff) / (n * alpha_new[k])
                # Sigma_fixed_point_new *= (p * np.trace(S_gaussian) / (p*np.trace(Sigma_fixed_point_new)))
                Sigma_fixed_point_new = S_left + S_right
                Sigma_fixed_point_new *= p / np.trace(Sigma_fixed_point_new)

                convergence_fp = True
                convergence_fp = convergence_fp and (math.sqrt(
                    np.inner(mu_fixed_point - mu_fixed_point_new, mu_fixed_point - mu_fixed_point_new) / p) < 10 ** (
                                                         -3))
                convergence_fp = convergence_fp and (
                            np.linalg.norm(Sigma_fixed_point_new - Sigma_fixed_point, ord='fro') / p) < 10 ** (-3)

                mu_fixed_point = mu_fixed_point_new.copy()

                if self.HDDA:
                    Sigma_fixed_point = HDDA_regularization(Sigma_fixed_point_new.copy(), regularizator=self.reg)
                else:
                    Sigma_fixed_point = Sigma_fixed_point_new.copy()

                ite_fp += 1
                s, u = scipy.linalg.eigh(Sigma_fixed_point, lower=True, check_finite=True)
                small_e = _eigvalsh_to_eps(s, None, None)
                if np.min(s) < -small_e:
                    print("min", np.min(s))
                    print("eps", small_e)
                    print('the input matrix must be positive semidefinite')
                    Sigma_fixed_point += (- np.min(s) + 50 * small_e) * np.eye(*Sigma_fixed_point.shape)

            mu_new[k] = mu_fixed_point
            Sigma_new[k] = Sigma_fixed_point #/ (np.trace(Sigma_fixed_point))

            # UPDATE tau
            # diff = self.X_hat - mu_new[k]
            # tau_new[:, k] = (np.dot(diff, np.linalg.inv(Sigma_new[k])) * diff).sum(1) / p
            # tau_new[:, k] = np.where(tau_new[:, k] < 10 ** (-12), 10 ** (-12),
            #                          np.where(tau_new[:, k] > 10 ** (12), 10 ** (12), tau_new[:, k]))

        return alpha_new, mu_new, Sigma_new, tau_new

    def fit(self, X):
        ''' Fit the data to the model running the F-EM algorithm

        Parameters
        ----------
        X: array-like, shape (n, p)
            data

        Returns
        ----------
        self
        '''

        n, p = X.shape

        self._initialize(X)

        convergence = False

        ite = 0

        while not (convergence) and ite < self.max_iter:

            # Compute conditional probabilities:
            # start = time.time()
            cond_prob = self._e_step(X)
            # print("estep", time.time() - start)
            # Update estimators:
            # start = time.time()
            alpha_new, mu_new, Sigma_new, tau_new = self._m_step(X, cond_prob)
            # print("mstep", time.time() - start)

            # Check convergence:
            # print('iteration', ite)
            if ite > 5:  # tol from fixed point should be bigger than general tolerance rate
                convergence = True
                k = 0
                while convergence and k < self.K:
                    convergence = convergence and math.sqrt(
                        np.inner(mu_new[k] - self.mu_[k], mu_new[k] - self.mu_[k]) / p) < 10 ** (-6)
                    convergence = convergence and (
                                (np.linalg.norm(Sigma_new[k] - self.Sigma_[k], ord='fro') / (p)) < 10 ** (-6))
                    convergence = convergence and (math.fabs(alpha_new[k] - self.alpha_[k]) < 10 ** (-6))
                    k += 1

            self.alpha_ = np.copy(alpha_new)
            self.mu_ = np.copy(mu_new)
            self.Sigma_ = np.copy(Sigma_new)
            ite += 1

        self.labels_ = np.array([i for i in np.argmax(cond_prob, axis=1)])
        self.n_iter_ = ite
        self.converged_ = convergence

        # Outlier rejection
        outlierness = np.zeros((n,)).astype(bool)

        if self.thres is None:
            self.thres = 0.05
        thres = chi2.ppf(1 - self.thres, p)

        for k in range(self.K):
            data_cluster = X[self.labels_ == k, :]
            diff_cluster = data_cluster - self.mu_[k]
            sig_cluster = np.mean(diff_cluster * diff_cluster)
            maha_cluster = (np.dot(diff_cluster, np.linalg.inv(self.Sigma_[k])) * diff_cluster).sum(1) / sig_cluster
            outlierness[self.labels_ == k] = (maha_cluster > thres)

        self.labels_[outlierness] = -1

        self.labels_ = self.labels_.astype(str)

        return (self)

    # NOT CODED YET / OLD VERSION VIOLETA
    # def predict(self, Xnew, thres=None):
    #
    #     n, p = Xnew.shape
    #
    #     cond_prob_matrix = np.zeros((n, self.K))
    #
    #     for k in range(self.K):
    #         psd = _PSD(self.Sigma_[k])
    #         prec_U, logdet = psd.U, psd.log_pdet
    #         diff = Xnew - self.mu_[k]
    #         sig = np.mean(diff * diff)
    #         maha = (np.dot(diff, np.linalg.inv(self.Sigma_[k])) * diff).sum(1)
    #         logdensity = -0.5 * (logdet + maha)
    #         cond_prob_matrix[:, k] = np.exp(logdensity) * self.alpha_[k]
    #
    #     sum_row = np.sum(cond_prob_matrix, axis=1)
    #     bool_sum_zero = (sum_row == 0)
    #
    #     cond_prob_matrix[bool_sum_zero, :] = self.alpha_
    #     cond_prob_matrix /= cond_prob_matrix.sum(axis=1)[:, np.newaxis]
    #
    #     new_labels = np.array([i for i in np.argmax(cond_prob_matrix, axis=1)])
    #
    #     outlierness = np.zeros((n,)).astype(bool)
    #
    #     if thres is None:
    #         thres = self.thres
    #     thres = chi2.ppf(1 - thres, p)
    #
    #     for k in range(self.K):
    #         data_cluster = Xnew[new_labels == k, :]
    #         diff_cluster = data_cluster - self.mu_[k]
    #         sig_cluster = np.mean(diff_cluster * diff_cluster)
    #         maha_cluster = (np.dot(diff_cluster, np.linalg.inv(self.Sigma_[k])) * diff_cluster).sum(1) / sig_cluster
    #         outlierness[new_labels == k] = (maha_cluster > thres)
    #
    #     new_labels[outlierness] = -1
    #
    #     new_labels = new_labels.astype(str)
    #
    #     return (new_labels)
