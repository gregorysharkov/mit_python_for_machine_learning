import time
import numpy as np
import scipy.sparse as sparse

ITER = 100
K = 10
N = 10000

def naive(indices, k):
		mat = [[1 if i == j else 0 for j in range(k)] for i in indices]
		return np.array(mat).T


def with_sparse(indices, k):
		n = len(indices)
		M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
		return M


Y = np.random.randint(0, K, size=N)

t0 = time.time()
for i in range(ITER):
		naive(Y, K)
print(time.time() - t0)


t0 = time.time()
for i in range(ITER):
		with_sparse(Y, K)
print(time.time() - t0)

print(sparse.coo_matrix([1]*5,(Y, range(5)))).toarray()


# x = np.array([
#     [2.5,0.5,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1],
#     [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9],
# ]).T

#     # [2.9,0.8,2.0,1.95,3.0,2.4,2.1,1.1,1.4,1.1]

# feature_means = x.mean(axis=0) #[1.81,1.91]

# x_centered = x - feature_means
# cov_matrix = np.cov(x_centered.T)
# eigen_values = np.linalg.eig(cov_matrix)[0]
# eigen_vectors = np.linalg.eig(cov_matrix)[1]

# print(np.dot(x_centered,eigen_vectors[:2,:].T))
# print(np.dot(x_centered,principal_components(x_centered)[:2,:].T))