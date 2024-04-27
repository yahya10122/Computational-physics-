
import math

def fnc(y, x):
    return (math.sin(x) + 2*y)

def heun(h, y_o, x_o, x1):
    x = [x_o]
    y = [y_o]

    for i in range(math.ceil((x1 - x_o) / h)):
        k1 = h * fnc(y[-1], x[-1])
        k2 = h * fnc(y[-1] + k1, x[-1] + h)
        y_next = y[-1] + (k1 + k2) / 2
        x_next = x[-1] + h
        x.append(x_next)
        y.append(y_next)

    return y[-1]

print(heun(0.02, 1, 0, 0.08))