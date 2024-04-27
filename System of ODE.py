import numpy as np
import math
def f(x, y_vec):
    """Computes the vector f"""
    f_vec = np.zeros(2) 
    f_vec[0] =  y_vec[1] # y1'=y2
    f_vec[1] = 2*y_vec[1]-2*y_vec[0]+ (math.exp(2*x))*math.sin(x) # y2'=2y2-2y1+e^2xsin(x)
    return f_vec

def iterate(x, y_n):
    """Computes Y(t_n+1)"""
    k1 = h * f(x, y_n)
    k2 = h * f(x + h / 2, y_n + k1 / 2)
    k3 = h * f(x + h / 2, y_n + k2 / 2)
    k4 = h * f(x + h, y_n + k3)
    y_nplus1 = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_nplus1

# Number of steps

# initial x
a = 0
x = a

# final x
b = 1
h = 0.1
# step size
N = (b - a) / h

# initial conditions
y = np.array([-0.4, -0.6])  # y1(0), y2(0)

# iterate N times
for i in range(math.ceil(N)):
    y = iterate(x, y)
    x = x + h

print(y)