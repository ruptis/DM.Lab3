import numpy as np
from scipy import stats


class GaussParameters:
    def __init__(self, mean, deviation):
        self.mean = mean
        self.deviation = deviation


def get_gauss_parameters(vector):
    return GaussParameters(np.mean(vector), np.std(vector))


def get_probability_density_function(vector, probability, arrange):
    gauss_parameters = get_gauss_parameters(vector)
    return stats.norm.pdf(arrange, gauss_parameters.mean, gauss_parameters.deviation) * probability


def generate_vectors(first_gauss_parameters, second_gauss_parameters, count):
    first_vector = get_random_vector(first_gauss_parameters, count)
    second_vector = get_random_vector(second_gauss_parameters, count)
    return first_vector, second_vector


def get_random_vector(gauss_parameters, count):
    return list(np.random.normal(gauss_parameters.mean, gauss_parameters.deviation, count))


def get_interval(first_vector, second_vector):
    all_points = first_vector + second_vector
    return min(all_points), max(all_points)


def get_intersection(first_function_values, second_function_values):
    return np.argwhere(np.diff(np.sign(first_function_values - second_function_values))).flatten()[0]


def get_errors(first_function_values, second_function_values, intersection, step):
    detection_error = np.trapz(first_function_values[intersection:], dx=step)
    false_positive = np.trapz(second_function_values[:intersection], dx=step)
    return detection_error, false_positive
