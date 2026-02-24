import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from scipy import integrate
from functools import partial
import time
from numba import njit

h = 2.04
theta = np.linspace(0, np.pi, 300)[:, np.newaxis]

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

@njit()
def modelJit(x, A, d, R, sig_para, sig_perp):
    theta = np.linspace(0, np.pi, 300)
    result = np.zeros(x.size)

    for i in range(x.size):
        y = x[i]

        sigTH = np.sqrt( sig_para**2*np.cos(theta)**2 + sig_perp**2*np.sin(theta)**2 )

        alpha = 2*( d*sig_perp*sig_para*np.sin(theta) / sigTH )**2
        beta = ( 2*sig_para**2*d*np.cos(theta) / sigTH**2 ) * y

        denom = 2.506628 * sigTH * (1 + R**2 + 2*R*np.exp(-2*d**2*sig_para**2))
        jp = np.exp( -y**2/(2*sigTH**2)) * (1 + R**2 + 2*R*np.exp(-alpha)*np.cos(beta)) / denom
        jp *= np.sin(theta)
        
        JBest = np.trapz(jp, theta)      # Integrate over theta
        result[i] = JBest

    # norm = np.abs(np.trapz(result, x))    #TODO: Failing on numba, don't know why
    # result /= norm 
    result *= A
    return result



# Model with Scipy integration
def modelQuad(y, A, d, R, sig_para, sig_perp):  # Numerical integration using trapz
    
    def integrand(theta, y):
        sigTH = np.sqrt( sig_para**2*np.cos(theta)**2 + sig_perp**2*np.sin(theta)**2 )

        alpha = 2*( d*sig_perp*sig_para*np.sin(theta) / sigTH )**2
        beta = ( 2*sig_para**2*d*np.cos(theta) / sigTH**2 ) * y

        denom = 2.506628 * sigTH * (1 + R**2 + 2*R*np.exp(-2*d**2*sig_para**2))
        jp = np.exp( -y**2/(2*sigTH**2)) * (1 + R**2 + 2*R*np.exp(-alpha)*np.cos(beta)) / denom
        jp *= np.sin(theta)
        return jp
    
    JBest = np.zeros(len(y))
    for i, v in enumerate(y):
        JBest[i] = integrate.quad(partial(integrand, y=v), 0, np.pi)[0]
    
    JBest /= np.abs(np.trapz(JBest, x=y))
    JBest *= A
    return JBest

defaultPars = {
    "A" : 1,
    "d" : 1,
    "sig_para" : 3,
    "sig_perp" : 5,
    "R" : 1
}

y = np.linspace(-20, 20, 100)
result = modelJit(y, **defaultPars)
np.random.seed(1)
noise = np.random.random(len(y)) - 0.5  # Half positive and half negative
data = result + noise * 0.02
error = noise * 0.04
# print(y, "\n", data, "\n", error)

plt.errorbar(y, data, error, fmt=".")
for model in [modelJit, modelTrpz, modelQuad]:
    t0 = time.time()

    costFun = cost.LeastSquares(y, data, error, model)
    m = Minuit(costFun, **defaultPars)
    m.simplex()
    m.migrad()

    print(m.covariance.correlation())

    leg = ""
    for p, v, e in zip(defaultPars, m.values, m.errors):
        leg += f"{p}: {v:.2f} +/- {e:.2f}\n"
    plt.plot(y, model(y, *m.values), label=leg)
    t1 = time.time()
    print(f"\nRunning time: {t1-t0:.2f} seconds\n")

plt.legend()
plt.show()
