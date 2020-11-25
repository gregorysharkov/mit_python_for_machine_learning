import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt(".\\netflix\\toy_data.txt")

# TODO: Your code here
# problem 1
# seeds = [0,1,2,3,4]
# Ks = [1,2,3,4]

# for i, K in enumerate(Ks):
#     k_best_mix, k_best_post, k_best_cost = None, None, np.inf
#     for seed in seeds:
#         init_mix, init_post = common.init(X, K, seed)
#         k_mix, k_post, k_cost= kmeans.run(X, init_mix, init_post)
#         if k_cost < k_best_cost:
#             k_best_mix, k_best_post, k_best_cost = k_mix, k_post, k_cost

#     #common.plot(X, k_best_mix, k_best_post, f"K={K}, cost={k_best_cost}")
#     print(f"best cost for K={K} is {k_best_cost}")

#problem2
# Ks = [1, 2, 3, 4]
# seeds = [0, 1, 2, 3, 4]
# BICs = np.empty(len(Ks))

# for i, K in enumerate(Ks):
#     k_best_mix, k_best_post, k_best_cost = None, None, np.inf
#     em_best_mix, em_best_post, em_best_ll = None, None, -np.inf
#     for seed in seeds:
#         init_mix, init_post = common.init(X, K, seed)
#         k_mix, k_post, k_cost= kmeans.run(X, init_mix, init_post)
#         em_mix, em_post, em_ll= em.run(X, init_mix, init_post)
#         if k_cost < k_best_cost:
#             k_best_mix, k_best_post, k_best_cost = k_mix, k_post, k_cost
#         if em_ll > em_best_ll:
#             em_best_mix, em_best_post, em_best_ll = em_mix, em_post, em_ll
#     BICs[i] = common.bic(X, em_best_mix, em_best_ll)
#     # print(f"Log likelyhood for K={K} is: {str(em_best_ll)}")
#     # common.plot(X, k_best_mix, k_best_post, "K-means K={}".format(K))
#     # common.plot(X, em_best_mix, em_best_post, "EM K={}".format(K))
    
# print("BICs: ", BICs)
# print("Best BIC: ", np.max(BICs))
# print("Best K: ", Ks[np.argmax(BICs)])

#last problem
X = np.loadtxt(".\\netflix\\netflix_incomplete.txt")

K = 1
seeds = [0, 1, 2, 3, 4]

em_best_mix, em_best_post, em_best_ll = None, None, -np.inf
for seed in seeds:
    init_mix, init_post = common.init(X, K, seed)
    em_mix, em_post, em_ll= em.run(X, init_mix, init_post)
    if em_ll > em_best_ll:
        em_best_mix, em_best_post, em_best_ll = em_mix, em_post, em_ll
print("K = {}, LL = {}".format(K, em_best_ll))

X_fill_pred = em.fill_matrix(X, em_best_mix)
X_fill = np.load(".\\netflix\\netflix_complete", allow_pickle=True)

print("X_filled Error:", common.rmse(X_fill_pred, X_fill))