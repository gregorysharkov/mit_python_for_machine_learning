import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt(".\\netflix\\toy_data.txt")

# TODO: Your code here
seeds = [0,1,2,3,4]
Ks = [1,2,3,4]

for i, K in enumerate(Ks):
    k_best_mix, k_best_post, k_best_cost = None, None, np.inf
    for seed in seeds:
        init_mix, init_post = common.init(X, K, seed)
        k_mix, k_post, k_cost= kmeans.run(X, init_mix, init_post)
        if k_cost < k_best_cost:
            k_best_mix, k_best_post, k_best_cost = k_mix, k_post, k_cost

    #common.plot(X, k_best_mix, k_best_post, f"K={K}, cost={k_best_cost}")
    print(f"best cost for K={K} is {k_best_cost}")
