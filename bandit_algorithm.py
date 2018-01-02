import operator
import math


class RandomAlgorithm:

    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.sum_results = 0

    def get_next_decision(self):
        return 0

    def process_decision_results(self, k, result):
        self.sum_results += result

    def get_sum_results(self):
        return self.sum_results


class GittinsIndexAlgorithm:
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.sum_results = 0
        # self.machines_estimated_probabilities = [0.0] * K
        self.machines_number_of_attempts = [0.0] * K
        self.machines_sum_of_results = [0.0] * K

    def get_machine_estimated_probability(self, k):
        number_of_attempt = self.machines_number_of_attempts[k]
        if number_of_attempt == 0:
            return 0.0
        else:
            return self.machines_sum_of_results[k] / number_of_attempt

    def get_next_decision(self):
        machines_estimated_probabilities = [self.get_machine_estimated_probability(k) for k in range(self.K)]
        index, _ = max(enumerate(machines_estimated_probabilities), key=operator.itemgetter(1))
        return index

    def process_decision_results(self, k, result):
        self.machines_number_of_attempts[k] += 1
        self.machines_sum_of_results[k] += result
        self.sum_results += result

    def get_sum_results(self):
        return self.sum_results


class ExplorationAlgorithm:

    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.t = 0
        self.sum_results = 0
        self.machines_number_of_attempts = [0.0] * K
        self.machines_sum_of_results = [0.0] * K

    def get_machine_estimated_probability(self, k):
        number_of_attempts = self.machines_number_of_attempts[k]
        if number_of_attempts == 0:
            return 0.0
        else:
            return self.machines_sum_of_results[k] / number_of_attempts

    def get_next_decision(self):
        if self.t < self.T / 2:
            return self.t % self.K
        else:
            machines_estimated_probabilities = [self.get_machine_estimated_probability(k) for k in range(self.K)]
            index, _ = max(enumerate(machines_estimated_probabilities), key=operator.itemgetter(1))
            return index

    def process_decision_results(self, k, result):
        self.machines_number_of_attempts[k] += 1
        self.machines_sum_of_results[k] += result
        self.sum_results += result
        self.t += 1

    def get_sum_results(self):
        return self.sum_results


class UCB1Algorithm:

    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.t = 0
        self.sum_results = 0
        self.machines_number_of_attempts = [0.0] * K
        self.machines_sum_of_results = [0.0] * K

    def get_ucb_optimist_probability(self, k):
        number_of_attempts = self.machines_number_of_attempts[k]
        if number_of_attempts == 0:
            return 0.0
        else:
            return (self.machines_sum_of_results[k] / number_of_attempts) + \
                   math.sqrt(math.log(self.t) / (2 * number_of_attempts))

    def get_next_decision(self):
        if self.t < self.K:
            return self.t
        else:
            machines_estimated_probabilities = [self.get_ucb_optimist_probability(k) for k in range(self.K)]
            index, _ = max(enumerate(machines_estimated_probabilities), key=operator.itemgetter(1))
            return index

    def process_decision_results(self, k, result):
        self.machines_number_of_attempts[k] += 1
        self.machines_sum_of_results[k] += result
        self.sum_results += result
        self.t += 1

    def get_sum_results(self):
        return self.sum_results


class KLUCBAlgorithm:

    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.t = 0
        self.sum_results = 0
        self.machines_number_of_attempts = [0.0] * K
        self.machines_sum_of_results = [0.0] * K

    def get_machine_estimated_probability(self, k):
        number_of_attempts = self.machines_number_of_attempts[k]
        if number_of_attempts == 0:
            return 0.0
        else:
            return self.machines_sum_of_results[k] / number_of_attempts

    def kl_distance(self, p1, p2):
        if p1 == 0.0:
            p1 = 1e-6
        if p2 == 0.0:
            p2 = 1e-6
        if p1 == 1.0:
            p1 = 1.0 - 1e-6
        if p2 == 1.0:
            p2 = 1.0 - 1e-6
        return p1 * math.log(p1 / p2) + (1 - p1) * math.log((1 - p1) / (1 - p2))

    def find_upper_bound_max_confidence(self, k):
        log_t = math.log(self.t)
        number_attempts = self.machines_number_of_attempts[k]
        estimated_probability = self.get_machine_estimated_probability(k)

        min_search, max_search = estimated_probability, 1.0
        while (max_search - min_search) > 1e-6:
            search_value = (min_search  + max_search) / 2.0
            if self.kl_distance(estimated_probability, search_value) <= (log_t / number_attempts):
                min_search, max_search = search_value, max_search
            else:
                min_search, max_search = min_search, search_value

        return min_search

    def get_next_decision(self):
        if self.t < self.K:
            return self.t
        else:
            upper_bounds_max_confidence = [self.find_upper_bound_max_confidence(k) for k in range(self.K)]
            index, _ = max(enumerate(upper_bounds_max_confidence), key=operator.itemgetter(1))
            return index

    def process_decision_results(self, k, result):
        self.machines_number_of_attempts[k] += 1
        self.machines_sum_of_results[k] += result
        self.sum_results += result
        self.t += 1

    def get_sum_results(self):
        return self.sum_results

