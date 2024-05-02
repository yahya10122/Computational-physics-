import numpy as np

def f(x, y_vec):
    """Computes the vector f"""
    y, z = y_vec
    f = np.zeros(2)
    f[0] = z
    f[1] = 6 * x
    return f

def rk4(x, y_vec, h):
    """Computes the next step using the fourth-order Runge-Kutta method"""
    k1 = h * f(x, y_vec)
    k2 = h * f(x + h / 2, y_vec + k1 / 2)
    k3 = h * f(x + h / 2, y_vec + k2 / 2)
    k4 = h * f(x + h, y_vec + k3)
    y_nplus1 = y_vec + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_nplus1

def solve_ode(y0, z0, t_span, h):
    """Solves the ODE using the Runge-Kutta method"""
    a, b = t_span
    x = a
    N = (b - a) / h
    y = y0
    z = z0
    for i in range(int(N)):
        y_vec = np.array([y, z])
        y_vec_next = rk4(x, y_vec, h)
        y, z = y_vec_next
        x += h
    return y, z

# Initial conditions
y0 = 2  # y(1) = 2
z0 = 2  # Initial guess for z(1)

# Time span
t_span = [1, 2]

# Step size
h = 0.1

# Tolerance
tol = 1e-6

# Boundary condition at t = 2
y_boundary = 9  # y(2) = 9

# Root-finding loop
lb = -10  # Lower bound for z(1)
ub = 10  # Upper bound for z(1)

while True:
    z0 = (lb + ub) / 2  # Midpoint as the new guess

    # Solve the ODE
    y_final, z_final = solve_ode(y0, z0, t_span, h)

    # Check if the boundary condition is satisfied
    if abs(y_final - y_boundary) < tol:
        break
    elif y_final < y_boundary:
        lb = z0  # Update the lower bound
    else:
        ub = z0  # Update the upper bound

# Print the final solution
print(f"Final solution: y(1) = {y0}, y(2) = {y_final:.6f}, y'(2) = {z_final:.6f}")
