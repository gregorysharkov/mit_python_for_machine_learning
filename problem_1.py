import numpy as np

w = np.array([[1,0],[0,1],[-1,0],[0,-1]])
w_0 = np.array([-1,-1,-1,-1]).T
v = np.array([[1,1,1,1],[-1,-1,-1,-1]])
v_0 = np.array([0,2]).T
x = np.array([3,14])

def calculate_z(x, w, w_0):
    '''
    function calculates z, given input and weights
    '''
    z = np.asarray((w.dot(x) + w_0))
    return z

def calculate_fz(z):
    '''
    function calculates f(z) based on given z
    '''
    fz_func = lambda x: np.max([x,0])
    fz = np.array([fz_func(x) for x in z])
    return fz

def calculate_softmax(fu):
    '''
    function calculates softmax probabilities
    '''
    denom = np.sum(np.exp(fu))
    o = np.exp(fu) / denom
    return o

z = calculate_z(x, w, w_0)
fz = calculate_fz(z)
print(fz)
u = calculate_z(fz, v, v_0)
fu = calculate_fz(u)
o = calculate_softmax(fu)
print(o)