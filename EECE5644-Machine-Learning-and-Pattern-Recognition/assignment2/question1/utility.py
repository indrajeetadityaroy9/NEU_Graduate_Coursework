import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve


def generate_data(N, pdf_params):
    n = pdf_params['mu'].shape[2]
    u = np.random.rand(N)

    thresholds = np.cumsum(np.append(
        pdf_params['gmm_a'][0].dot(pdf_params['priors'][0]),
        pdf_params['gmm_a'][1].dot(pdf_params['priors'][1])
    ))
    thresholds = np.insert(thresholds, 0, 0)

    labels = np.zeros(N, dtype=int)
    x = np.zeros((N, n))

    for l in range(2):
        for i in range(2):
            lower_bound = np.sum(pdf_params['gmm_a'][l, :i]) * pdf_params['priors'][l] + np.sum(
                pdf_params['priors'][:l])
            upper_bound = np.sum(pdf_params['gmm_a'][l, :i + 1]) * pdf_params['priors'][l] + np.sum(
                pdf_params['priors'][:l])

            indices = np.argwhere((lower_bound <= u) & (u < upper_bound)).flatten()

            x[indices, :] = multivariate_normal.rvs(mean=pdf_params['mu'][l, i, :], cov=pdf_params['Sigma'][l, i, :, :],
                                                    size=len(indices))

            labels[indices] = l

    return x, labels


def bayesian_optimal_classifier(x, pdf_params):
    likelihood_0 = (pdf_params['gmm_a'][0, 0] * multivariate_normal.pdf(x, pdf_params['mu'][0, 0],
                                                                        pdf_params['Sigma'][0, 0]) +
                    pdf_params['gmm_a'][0, 1] * multivariate_normal.pdf(x, pdf_params['mu'][0, 1],
                                                                        pdf_params['Sigma'][0, 1]))
    likelihood_1 = (pdf_params['gmm_a'][1, 0] * multivariate_normal.pdf(x, pdf_params['mu'][1, 0],
                                                                        pdf_params['Sigma'][1, 0]) +
                    pdf_params['gmm_a'][1, 1] * multivariate_normal.pdf(x, pdf_params['mu'][1, 1],
                                                                        pdf_params['Sigma'][1, 1]))

    # Calculate posterior probabilities using Bayes' theorem
    posterior_0 = likelihood_0 * pdf_params['priors'][0]
    posterior_1 = likelihood_1 * pdf_params['priors'][1]

    # Decide the class based on which posterior probability is greater
    return np.where(posterior_1 > posterior_0, 1, 0), posterior_1 / (posterior_1 + posterior_0)


def empirical_min_p_error(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    min_p_error = 1
    min_p_error_point = (0, 0)
    min_p_error_gamma = 0
    for i, threshold in enumerate(thresholds):
        error = (fpr[i] * sum(labels == 0) + (1 - tpr[i]) * sum(labels == 1)) / len(labels)
        if error < min_p_error:
            min_p_error = error
            min_p_error_point = (fpr[i], tpr[i])
            min_p_error_gamma = threshold
    return min_p_error, min_p_error_point, min_p_error_gamma


def logistic_func(z):
    return 1 / (1 + np.exp(-z))


def nll(w, x, y):
    z = np.dot(x, w)
    return -np.sum(y * z - np.log(1 + np.exp(z)))


def compute_logistic_params(x, y):
    theta0 = np.zeros(x.shape[1])
    result = minimize(lambda w: nll(w, x, y), theta0, method='BFGS')
    return result.x


def classify(x, w):
    probabilities = logistic_func(np.dot(x, w))
    return np.where(probabilities > 0.4, 1, 0)


def error_rate(predictions, labels):
    return np.mean(predictions != labels)
