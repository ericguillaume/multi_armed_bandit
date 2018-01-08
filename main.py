from casino_machines import *
from bandit_algorithm import *


if __name__ == "__main__":
    K = 10
    T = 2000 # 10000
    NB_SCENARIOS = 1 #100

    sum_results = 0
    for _ in range(NB_SCENARIOS):
        casino = CasinoMachines(K)
        bandit = HomeMadeBayesianOptimistAlgorithm(K, T)

        for _ in range(T):
            k = bandit.get_next_decision()
            result = casino.run(k)
            bandit.process_decision_results(k, result)

        sum_results += bandit.get_sum_results()

    maximum_money_earned = NB_SCENARIOS * T * casino.get_max_probability()
    print("We have won {} euros out of a maximum of {}, i.e. {} %"
          .format(sum_results, maximum_money_earned, sum_results / maximum_money_earned))


