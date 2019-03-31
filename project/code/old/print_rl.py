import matplotlib.pyplot as plt
import json

with open("cum_regrets_greedy.txt", "r") as f:
    greedy_regret = json.load(f)

with open("cum_regrets_eps_greedy_1.txt", "r") as f:
    eps_greedy_1_regret = json.load(f)

with open("cum_regrets_eps_greedy_5.txt", "r") as f:
    eps_greedy_5_regret = json.load(f)


with open("cum_regrets_bayes.txt", "r") as f:
    bayes_regret = json.load(f)

plt.plot(greedy_regret, label="Greedy")
plt.plot(eps_greedy_1_regret, label="Eps-Greedy (1%)")
plt.plot(eps_greedy_5_regret, label="Eps-Greedy (5%)")
plt.plot(bayes_regret, label="Bayes")
plt.yscale("log")
plt.legend()
plt.show()
