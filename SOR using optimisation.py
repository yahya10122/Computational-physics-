import numpy as np

def norm(array1, array2):
    return np.linalg.norm(array1 - array2)

# Initialize variables
xo = np.array([[0], [0], [0]], dtype=float)
p = np.array([[0], [0], [0]], dtype=float)
po = np.array([[0], [0], [0]], dtype=float)
p_1 = np.array([[0], [0], [0]], dtype=float)
po_1 = np.array([[0], [0], [0]], dtype=float)
B = np.array([[24], [30], [-24]], dtype=float)
A = np.array([[4, 3, 0],
              [3, 4, -1],
              [0, -1, 4]], dtype=float)
n = 3
count = 0
count1 = 0
count3 = 0
w=1
# Perform Gauss-Seidel iterations:
while True:
    if count == 21:
        p = np.copy(xo)
    if count == 20:
        p_1 = np.copy(xo)
    if count1 == 35:
        po = np.copy(xo)
    if count1 ==36:
        po_1 = np.copy(xo)
        break
    x1 = np.copy(xo)  # Initialize x1 with xo
    for i in range(n):
        sum1 = np.dot(A[i, :i], x1[:i]) + np.dot(A[i, i+1:], xo[i+1:])
        x1[i] = w * ((B[i] - sum1) / A[i, i]) + (1-w)*xo[i]
    e = norm(xo, x1)
    xo = x1  # Update xo for the next iteration
    count += 1
    count1 +=1
# Calculate w
w = 2 / (1 + (1 - ((norm(po, xo) / norm(p, p_1))**(1/25)))**0.5)

xo = np.array([[0], [0], [0]], dtype=float)
# Continue with SOR iterations
while True:
    x1 = np.copy(xo)
    for i in range(n):
        sum1 = np.dot(A[i, :i], x1[:i]) + np.dot(A[i, i+1:], xo[i+1:])
        x1[i] = w * ((B[i] - sum1) / A[i, i]) + (1 - w) * xo[i]
    e = norm(xo, x1)
    xo = x1
    count3 += 1
    if e < 0.000001:
        break

print("Solution (x):")
print(xo)
print("Number of iterations:", count3)
