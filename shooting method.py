import numpy as np
import math

def f(x, y_vec):
    """Computes the vector f"""
    f_vec = np.zeros(2)
    f_vec[0] = y_vec[1]
    f_vec[1] = -3*y_vec[0]*y_vec[1]
    return f_vec

def iterate(x, y_n):
    """Computes Y(t_n+1)"""
    k1 = h * f(x, y_n)
    k2 = h * f(x + h / 2, y_n + k1 / 2)
    k3 = h * f(x + h / 2, y_n + k2 / 2)
    k4 = h * f(x + h, y_n + k3)
    y_nplus1 = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_nplus1

def solve_ode(y0, t_span, h):
    """Solves the ODE using the iterate function"""
    a, b = t_span
    x = a
    N = (b - a) / h
    y = y0.copy()

    for i in range(math.ceil(N)):
        y = iterate(x, y)
        x += h

    return y

# Initial conditions
y0 = np.array([0, 0.5])  # y(0) = 0, y'(0) = 0.5 (initial guess)

# Time span
t_span = [0, 2]

# Step size
h = 0.01

# Tolerance
tol = 1e-9

# Solve the ODE
y_final = solve_ode(y0, t_span, h)

# Print the initial solution
print(f"Initial solution: y(0) = 0, y(2) = {y_final[0]:.6f}, y'(2) = {y_final[1]:.6f}")

# Root-finding loop
lb = 1  # Lower bound for y'(0)
ub = 2  # Upper bound for y'(0)
b = 1 #(condition at the boundary)
while True:
    y0[1] = (lb + ub) / 2  # Midpoint as the new guess

    # Solve the ODE
    y_final = solve_ode(y0, t_span, h)

    # Check if the boundary condition is satisfied
    if abs(y_final[0] - b) < tol:
        break
    elif y_final[0] < 1:
        lb = y0[1]  # Update the lower bound
    else:
        ub = y0[1]  # Update the upper bound

# Print the final solution
print(f"Final solution: y(0) = 0, y(2) = {y_final[0]:.6f}, y'(2) = {y_final[1]:.6f}")