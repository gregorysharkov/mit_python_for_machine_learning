import numpy as np

w_fh = 0
w_fx = 0
b_f = -100
w_ch = -100
w_ih = 0
w_ix = 100
b_i = 100
w_cx = 50
w_oh = 0
w_ox = 100
b_o = 0
b_c = 0

x = np.array([0,0,1,1,1,0])
x_alt = np.array([1,1,0,1,1])

def sigmoid(x):
    return 1/(1+np.exp(-x))

h_zero = 0
c_zero = 0

h_output = []
for x_t in x_alt:
    #calculate forget state
    f_t = sigmoid(w_fh*h_zero + w_fx*x_t + b_f)
    print(f"f_t = {f_t}")
    #calculate input gate
    i_t = sigmoid(w_ih*h_zero + w_ix*x_t + b_i)
    print(f"i_t = {i_t}")
    #calculate output gate
    o_t = sigmoid(w_oh*h_zero + w_ox*x_t + b_c)
    print(f"o_t = {o_t}")
    #calculate cell gate
    c_t = f_t + i_t*np.tanh(w_ch*h_zero + w_cx*x_t + b_c)
    print(f"c_t = {c_t}")

    h_zero = o_t * np.tanh(c_t)
    h_output.append(np.round(h_zero, decimals=0))

print(h_output)