import numpy as np
from scipy.stats import norm

x = np.array([-1.0, 0.0, 4.0, 5.0, 6.0])
theta = np.array([.5,.5,6.0,7.0,1.0,4.0])

def calculate_likelyhood(x, theta):
    likelyhood = np.log(theta[0]*norm(loc=theta[2], scale=theta[4]).pdf(x) + theta[1]*norm(loc=theta[3], scale=theta[5]).pdf(x))
    print(x, theta[0]*norm(loc=theta[2], scale=theta[4]).pdf(x) + theta[1]*norm(loc=theta[3], scale=theta[5]).pdf(x))
    return likelyhood

print(
    np.sum(
        [calculate_likelyhood(el, theta) for el in x]
    )
)
