import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve
from scipy.optimize import minimize


def generate_data(N, pdf_params):
    # Generate synthetic data based on provided Gaussian Mixture Model (GMM) parameters

    n = pdf_params['mu'].shape[2]  # Dimension of the feature space
    u = np.random.rand(N)  # Uniform random values for determining class and component membership

    # Calculate thresholds for deciding class and component membership
    thresholds = np.cumsum(np.append(
        pdf_params['gmm_a'][0].dot(pdf_params['priors'][0]),
        pdf_params['gmm_a'][1].dot(pdf_params['priors'][1])
    ))
    thresholds = np.insert(thresholds, 0, 0)

    labels = np.zeros(N, dtype=int)  # Initialize labels array
    x = np.zeros((N, n))  # Initialize feature array

    # Assigning points to Gaussian components and classes
    for l in range(2):  # Two classes
        for i in range(2):  # Two components per class
            # Bounds for deciding membership to a particular class and component
            lower_bound = np.sum(pdf_params['gmm_a'][l, :i]) * pdf_params['priors'][l] + np.sum(pdf_params['priors'][:l])
            upper_bound = np.sum(pdf_params['gmm_a'][l, :i + 1]) * pdf_params['priors'][l] + np.sum(pdf_params['priors'][:l])
            # Indices of data points belonging to the current class and component
            indices = np.argwhere((lower_bound <= u) & (u < upper_bound)).flatten()
            # Sampling data points from the corresponding Gaussian distribution
            x[indices, :] = multivariate_normal.rvs(mean=pdf_params['mu'][l, i, :], cov=pdf_params['sigma'][l, i, :, :], size=len(indices))
            # Assigning class labels
            labels[indices] = l

    return x, labels


def bayesian_optimal_classifier(x, pdf_params):
    # Classify data points using Bayesian optimal classifier

    # Calculate likelihoods for each class
    likelihood_0 = (pdf_params['gmm_a'][0, 0] * multivariate_normal.pdf(x, pdf_params['mu'][0, 0],
                    pdf_params['sigma'][0, 0]) + pdf_params['gmm_a'][0, 1] * multivariate_normal.pdf(x, pdf_params['mu'][0, 1], pdf_params['sigma'][0, 1]))
    likelihood_1 = (pdf_params['gmm_a'][1, 0] * multivariate_normal.pdf(x, pdf_params['mu'][1, 0],
                    pdf_params['sigma'][1, 0]) + pdf_params['gmm_a'][1, 1] * multivariate_normal.pdf(x, pdf_params['mu'][1, 1], pdf_params['sigma'][1, 1]))
    # Calculate posterior probabilities using Bayes' theorem
    posterior_0 = likelihood_0 * pdf_params['priors'][0]
    posterior_1 = likelihood_1 * pdf_params['priors'][1]
    # Decide the class based on which posterior probability is greater
    return np.where(posterior_1 > posterior_0, 1, 0), posterior_1 / (posterior_1 + posterior_0)


def empirical_min_p_error(scores, labels):
    # Compute the empirical minimum probability of error based on ROC curve

    fpr, tpr, thresholds = roc_curve(labels, scores)  # Calculate ROC curve
    min_p_error = 1  # Initialize minimum error
    min_p_error_point = (0, 0)  # Initialize point at which minimum error occurs
    min_p_error_gamma = 0  # Initialize threshold (gamma) value at minimum error

    # Iterate over all thresholds to find the minimum error
    for i, threshold in enumerate(thresholds):
        error = (fpr[i] * sum(labels == 0) + (1 - tpr[i]) * sum(labels == 1)) / len(labels)
        if error < min_p_error:
            min_p_error = error
            min_p_error_point = (fpr[i], tpr[i])
            min_p_error_gamma = threshold

    return min_p_error, min_p_error_point, min_p_error_gamma


def logistic_func(z):
    # Logistic function for logistic regression
    return 1 / (1 + np.exp(-z))


def nll(w, x, y):
    # Negative log-likelihood function for logistic regression
    z = np.dot(x, w)
    return -np.sum(y * z - np.log(1 + np.exp(z)))


def compute_logistic_params(x, y):
    # Compute parameters for logistic regression using BFGS optimization
    theta0 = np.zeros(x.shape[1])  # Initial parameter guess
    result = minimize(lambda w: nll(w, x, y), theta0, method='BFGS')
    return result.x  # Return optimized parameters


def classify(x, w):
    # Classify data points using logistic regression model
    probabilities = logistic_func(np.dot(x, w))  # Calculate probabilities
    return np.where(probabilities > 0.4, 1, 0)  # Thresholding at 0.4


def error_rate(predictions, labels):
    # Compute error rate of predictions
    return np.mean(predictions != labels)  # Average of incorrect predictions
