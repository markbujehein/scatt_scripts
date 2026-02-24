import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
import time
plt.style.use("seaborn-poster")

Ns = [20, 50, 100, 200, 300, 400, 500, 700, 1000]
fitPars = []
times = []

for N in Ns:
    h = 2.04
    theta = np.linspace(0, np.pi, N)[:, np.newaxis]

    def modelTrpz(x, A, d, R, sig_para, sig_perp):  # Numerical integration using trapz
        y = x[np.newaxis, :]

        sigTH = np.sqrt( sig_para**2*np.cos(theta)**2 + sig_perp**2*np.sin(theta)**2 )

        alpha = 2*( d*sig_perp*sig_para*np.sin(theta) / sigTH )**2
        beta = ( 2*sig_para**2*d*np.cos(theta) / sigTH**2 ) * y

        denom = 2.506628 * sigTH * (1 + R**2 + 2*R*np.exp(-2*d**2*sig_para**2))
        jp = np.exp( -y**2/(2*sigTH**2)) * (1 + R**2 + 2*R*np.exp(-alpha)*np.cos(beta)) / denom
        jp *= np.sin(theta)

        JBest = np.trapz(jp, x=theta, axis=0)
        JBest /= np.abs(np.trapz(JBest, x=y))
        JBest *= A
        return JBest

    
    defaultPars = {"A":1, "d":1, "R":1, "sig_para":3, "sig_perp":5}  # TODO: Starting parameters and bounds?
    
    y = np.linspace(-20, 20, 100)
    result = modelTrpz(y, **defaultPars)
    np.random.seed(11)
    noise = np.random.random(len(y)) - 0.5  # Half positive and half negative
    data = result + noise * 0.02
    error = noise * 0.04

    t0 = time.time()

    costFun = cost.LeastSquares(y, data, error, modelTrpz)
    m = Minuit(costFun, **defaultPars)
    m.simplex()
    m.migrad()
    fitPars.append(m.values)

    t1 = time.time()
    times.append(t1-t0)

fitPars = np.array(fitPars).T

relDiffs = (fitPars - fitPars[:, 0][:, np.newaxis]) / fitPars[:, 0][:, np.newaxis]
for pars, key in zip(relDiffs, defaultPars):
    plt.plot(Ns, pars, "-o", label=key)
plt.title("Relative difference to first point")
plt.xlabel("Number of integration points")
plt.legend()
plt.show()

relDiffs = (fitPars[:, :-1] - fitPars[:, 1:]) / fitPars[:, :-1]
for pars, key in zip(relDiffs, defaultPars):
    plt.plot(Ns[1:], pars, "-o", label=key)
plt.title("Relative difference to previous point")
plt.xlabel("Number of integration points")
plt.legend()
plt.show()

# plt.plot(Ns, times, label="Times")
# plt.legend()
# plt.show()
