from numpy.random import uniform, binomial


class CasinoMachines():

    def __init__(self, K):
        self.K = K  # number of machines
        self.machines_win_probas = [uniform(0.01, 0.2) for _ in range(K)]

    def run(self, k):
        return binomial(1, self.machines_win_probas[k])

    def get_max_probability(self):
        return max(self.machines_win_probas)


class MockedCasinoMachines():

    def __init__(self, K, results):
        self.t = 0
        self.K = K  # number of machines
        self.machines_win_probas = [uniform(0.01, 0.2) for _ in range(K)]
        self.results = results

    def run(self, k):
        result = self.results[self.t]
        self.t += 1
        return result

    def get_max_probability(self):
        return max(self.machines_win_probas)


