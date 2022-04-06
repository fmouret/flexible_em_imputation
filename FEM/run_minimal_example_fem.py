import numpy as np
import matplotlib.pyplot as plt
from _fem_missing_data_vstable import FEM
import time
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import random
import scipy


def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal


def generate_synthetic_dataset(n_samples=2000, n_features=100, K=3, data_type='Gaussian', plot_generated_data=False):

    Mu_all = np.zeros([n_features, K])
    S_all = np.zeros([n_features, n_features, K])

    Mix = np.ones(K) / K
    Mix = Mix / Mix.sum()

    feat = np.zeros([n_samples, n_features])
    idx_sample_init = 0

    for k in range(K):

        Mu_all[:, k] = np.random.uniform(0, 1, size=(1, n_features))
        sigma_2 = random.uniform(0.0005, 0.005)
        c = random.uniform(0.1, 0.9)

        vector_var = np.zeros(n_features)
        for idx_var in range(n_features):
            vector_var[idx_var] = sigma_2 / (1- c**2) * (c ** idx_var)
        S_all[:, :, k] = scipy.linalg.toeplitz(vector_var)

        if data_type == 'Gaussian':
            feat[idx_sample_init: idx_sample_init + int(n_samples * Mix[k]), :] = scipy.stats.multivariate_normal(
                mean=Mu_all[:, k], cov=S_all[:, :, k]).rvs(int(n_samples * Mix[k]))
            print("generating gaussian data")
        elif data_type == 'Student':
            feat[idx_sample_init: idx_sample_init + int(n_samples * Mix[k]), :] = multivariate_t_rvs(
                m=Mu_all[:, k], S=S_all[:, :, k], df=5, n=int(n_samples * Mix[k]))
        else:
            raise ValueError('Wrong data type: Gaussian or Student ')

        idx_sample_init += int(n_samples * Mix[k])

    max_r = 100
    min_r = 1
    feat = (feat - feat.min()) / (np.percentile(feat, 98) - feat.min()) * (max_r - min_r) + min_r

    feat = feat + feat.min()

    if plot_generated_data:
        idx_sample_init = 0
        for k in range(K):
            plt.plot(feat[idx_sample_init: idx_sample_init + int(n_samples * Mix[k]), :].T, color=cycle_colors[k])
            idx_sample_init += int(n_samples * Mix[k])
        plt.grid(linestyle=':')
        plt.show()

    return feat


if __name__ == '__main__':

    K = 3
    n_iter = 50
    percentage_missing = 0.5

    data_type = "Student"
    data_type = "Gaussian"

    # synthetic dataset
    Norm = MinMaxScaler(feature_range=(0, 1))
    list_mae_mean = []
    list_mae_knn = []
    list_mae_mice = []
    list_mae_gmm = []
    list_mae_rgmm = []
    list_mae_fem = []
    list_mae_fem_test = []

    list_mape_mean = []
    list_mape_knn = []
    list_mape_mice = []
    list_mape_gmm = []
    list_mape_rgmm = []
    list_mape_fem = []
    list_mape_fem_test = []

    for exp in range(n_iter):
        feat = generate_synthetic_dataset(K=K, n_samples=2000, n_features=10, data_type=data_type)
        n_missing_data = int(feat.shape[0] * feat.shape[1] * percentage_missing)

        print("ITER ", exp)
        feat_original_norm = Norm.fit_transform(feat)
        feat_original_norm = feat.copy()
        n_features = feat.shape[1]

        expe_is_correct = False
        while expe_is_correct is False:
            feat_array_nans = feat.copy()
            feat_array_nans.ravel()[np.random.choice(feat_original_norm.size, n_missing_data, replace=False)] = np.nan
            sum_nans = np.sum(np.isnan(feat_array_nans), axis=1)
            all_nans = np.where(sum_nans == feat_array_nans.shape[1])[0]
            # print("test")
            # print(all_nans)
            # print(len(all_nans) == 0)
            if len(all_nans) == 0:
                expe_is_correct = True
        feat_array_nans = Norm.fit_transform(feat_array_nans)

        # X_noise = np.random.uniform(0, 1, size=(100, feat_original_norm.shape[1]))
        # feat_original_norm = np.vstack([feat_original_norm, X_noise])
        # feat_array_nans = np.vstack([feat_array_nans, X_noise])

        Missing_data = np.isnan(feat_array_nans)

        print('MEAN')
        clf = SimpleImputer()
        feat_imputed = clf.fit_transform(feat_array_nans)
        feat_imputed = Norm.inverse_transform(feat_imputed)

        mae = mean_absolute_error(y_true=feat_original_norm[Missing_data],
                                  y_pred=feat_imputed[Missing_data])
        mape = mean_absolute_percentage_error(y_true=feat_original_norm[Missing_data],
                                              y_pred=feat_imputed[Missing_data])
        print(mae)
        print(mape)
        list_mae_mean.append(mae)
        list_mape_mean.append(mape)

        print("KNN")
        clf = KNNImputer()
        feat_imputed = clf.fit_transform(feat_array_nans)
        feat_imputed = Norm.inverse_transform(feat_imputed)

        mae = mean_absolute_error(y_true=feat_original_norm[Missing_data],
                                  y_pred=feat_imputed[Missing_data])
        mape = mean_absolute_percentage_error(y_true=feat_original_norm[Missing_data],
                                              y_pred=feat_imputed[Missing_data])
        print(mae)
        print(mape)
        list_mae_knn.append(mae)
        list_mape_knn.append(mape)

        print("MICE")
        clf = IterativeImputer(max_iter=20,
                               # estimator=estimator,
                               # n_nearest_features=20,
                               # sample_posterior= True,
                               skip_complete=True,
                               )
        feat_imputed = clf.fit_transform(feat_array_nans)
        feat_imputed = Norm.inverse_transform(feat_imputed)

        mae = mean_absolute_error(y_true=feat_original_norm[Missing_data],
                                  y_pred=feat_imputed[Missing_data])
        mape = mean_absolute_percentage_error(y_true=feat_original_norm[Missing_data],
                                              y_pred=feat_imputed[Missing_data])
        print(mae)
        print(mape)
        list_mae_mice.append(mae)
        list_mape_mice.append(mape)

        print("FEM")
        start = time.time()
        fem = FEM(K=K, max_iter=30, max_iter_fp=20, reg=1e-6,
                  HDDA=False)
        fem.fit(feat_array_nans)
        print("time", time.time() - start)
        feat_imputed = fem.X_hat
        feat_imputed = Norm.inverse_transform(feat_imputed)

        mae = mean_absolute_error(y_true=feat_original_norm[Missing_data],
                                  y_pred=feat_imputed[Missing_data])
        mape = mean_absolute_percentage_error(y_true=feat_original_norm[Missing_data],
                                              y_pred=feat_imputed[Missing_data])
        print(mae)
        print(mape)
        list_mae_fem.append(mae)
        list_mape_fem.append(mape)

    my_dict = {#'MEAN': list_mae_mean,
               'KNN': list_mae_knn,
               'MICE': list_mae_mice,
               'FEM': list_mae_fem,
               }

    extension = "_K_" + str(K) + "_iter_" + str(n_iter) + "_"
    np.save("results_synthetic_data_MAE_" + extension + ".npy", my_dict)
    fig, ax = plt.subplots(dpi=200)
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    ax.set_ylabel('MAE: reconstruction error')
    # ax.set_xlabel('N features:' + str(n_features) + ', GMM: ' + str(K) + ', N samples:' + str(n_samples) + ', N missing components:' + str(n_missing_data))
    ax.grid(linestyle=':')
    plt.show()

    my_dict = {  # 'MEAN': list_mae_mean,
        'KNN': list_mape_knn*100,
        'MICE': list_mape_mice*100,
        'FEM': list_mape_fem,
    }
    extension = "_K_" + str(K) + "_iter_" + str(n_iter) + "_" + "_dtype_" + data_type + '_'
    np.save("results_synthetic_data_MAPE_" + extension + ".npy", my_dict)

    fig, ax = plt.subplots(dpi=200)
    ax.boxplot(my_dict.values())
    ax.set_xticklabels(my_dict.keys())
    ax.set_ylabel('MAPE')
    # ax.set_xlabel('N features:' + str(n_features) + ', GMM: ' + str(K) + ', N samples:' + str(n_samples) + ', N missing components:' + str(n_missing_data))
    ax.grid(linestyle=':')
    plt.show()
