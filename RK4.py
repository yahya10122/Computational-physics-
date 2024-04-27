

import math

def fnc(y, x):
    return (x**3)

def RK4(h, y_o, x_o, x1):
    x = [x_o]
    y = [y_o]

    for i in range(math.ceil((x1 - x_o) / h)):
        m1 = fnc(y[-1], x[-1])
        m2 = fnc(y[-1] + 1/2 * m1*h, x[-1] + 1/2*h)
        m3 = fnc(y[-1]+m2*h/2,x[-1]+h/2)
        m4 = fnc(y[-1]+m3*h,x[-1]+h)
        y_next = y[-1] + h*(m1 + 2*m2+2*m3+m4) / 6
        x_next = x[-1] + h
        x.append(x_next)
        y.append(y_next)

    return y

print(RK4(0.2, 0, 0, 0.4))