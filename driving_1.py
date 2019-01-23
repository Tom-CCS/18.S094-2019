import numpy as np
import numpy.random as nr
import matplotlib.pyplot as pl
def linreg(xn, yn, max_iter = 10, w = np.zeros(2), alpha = 0.01):
    a = w[0]
    b = w[1]
    l = len(xn)
    for it in range(max_iter):
       f = lambda x: a * x + b
       grada = 0
       gradb = 0
       diff = 0
       for i in range(l):
           grada += (yn[i] - f(xn[i])) * xn[i];
           gradb += (yn[i] - f(xn[i]))
           diff += (yn[i] - f(xn[i])) * (yn[i] - f(xn[i]))
       a = a + alpha * grada
       b = b + alpha * gradb
       if (it % (max_iter // 10) == 0):
           print(diff / l)
    f = lambda x: a * x + b
    return f
pl.rcParams['figure.figsize'] = (8.0, 5.0)
N = 100
x = nr.rand(N)
err = [nr.normal(0, 1) for _ in range(N)]
a0 = nr.normal(2, 1)
b0 = nr.normal(3, 1)
f0 = lambda x: a0 * x + b0
y = f0(x) + err
fig = pl.figure()
figa = pl.gca()
f = linreg(x, y)
pl.plot(x, f(x), 'r--', label = 'linreg')
for i in range(N):
    if (f(x[i]) < y[i]):
        pl.plot(x[i], y[i], 'go')
    else:
        pl.plot(x[i], y[i],'bo')
pl.legend()
pl.show()



    
    
