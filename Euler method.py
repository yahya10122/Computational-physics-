import math

def fnc(y, x):
    return (x**2+y**2)
#h = dx, y_o = intial value of y, x_o = intial value of x, x1 = value of y you need to find at x1
def euler(h, y_o, x_o, x1):
    x = [x_o]
    y = [y_o]

    for i in range(math.ceil((x1 - x_o) / h)):
        x_next = x[-1] + h
        y_next = y[-1] + h * fnc(y[-1], x[-1])
        x.append(x_next)
        y.append(y_next)

    return y[-1]

print(euler(0.2, 0, 0, 0.4))