
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost
from iminuit.util import describe, make_func_code
from pathlib import Path
from mantid.simpleapi import Load, CropWorkspace, CloneWorkspace
from scipy import optimize
from scipy import signal
import time


repoPath = Path(__file__).absolute().parent
nbkwds = {
    "parallel" : False,
    "fastmath" : True
}

def main(nSpec, nGroups, showPlots):

    # joyPath = repoPath / "wsJOYsmall.nxs"
    # resPath = repoPath / "wsResSmall.nxs"
    joyPath = repoPath / "wsDHMTjoy.nxs"
    resPath = repoPath / "wsDHMTres.nxs"
    ipPath = repoPath / "ip2018_3.par"  
 
    wsJoY = Load(str(joyPath), OutputWorkspace="wsJoY")
    wsRes = Load(str(resPath), OutputWorkspace="wsRes")
    print("No of spec raw: ", wsRes.getNumberHistograms())

    firstSpec = 135
    lastSpec = 135+nSpec-1 #182

    firstIdx = firstSpec - 135
    lastIdx = lastSpec - 135

    wsJoY = CropWorkspace(InputWorkspace="wsJoY", OutputWorkspace="wsJoY", StartWorkspaceIndex=firstIdx, EndWorkspaceIndex=lastIdx)
    wsRes = CropWorkspace(InputWorkspace="wsRes", OutputWorkspace="wsRes", StartWorkspaceIndex=firstIdx, EndWorkspaceIndex=lastIdx)
    print("No of spec cropped: ", wsRes.getNumberHistograms())

    values, errors = fitGlobalFit(wsJoY, wsRes, False, ipPath, firstSpec, lastSpec, nGroups, showPlots)
    if showPlots:
        plt.show()
    return values, errors


def fitGlobalFit(ws, wsRes, gaussFlag,  InstrParsPath, firstSpec, lastSpec, nGroups, showPlots):

    dataX, dataY, dataE, dataRes, instrPars = extractData(ws, wsRes, InstrParsPath, firstSpec, lastSpec)
    dataX, dataY, dataE, dataRes, instrPars = takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars)

    print(f"\nNumber of groups: {nGroups}")
    idxList = groupDetectors(instrPars, nGroups, showPlots)
    dataX, dataY, dataE, dataRes = avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList)

    # TODO: Possible symetrisation goes here

    model, defaultPars, sharedPars = selectModelAndPars(gaussFlag)   
 
    totCost = 0
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
        totCost += calcCostFun(model, i, x, y, yerr, res, sharedPars)
    
    print("\nGlobal Fit Parameters:\n", describe(totCost))
    print("\nRunning Global Fit ...\n")

    initPars = {}
    # Populate with initial shared parameters
    for sp in sharedPars:
        initPars[sp] = defaultPars[sp]
    # Add initial unshared parameters
    unsharedPars = [key for key in defaultPars if key not in sharedPars]
    for up in unsharedPars:
        for i in range(len(dataY)):
            initPars[up+str(i)] = defaultPars[up]

    m = Minuit(totCost, **initPars)

    for i in range(len(dataY)):     # Limit for both Gauss and Gram Charlier
        m.limits["A"+str(i)] = (0, np.inf)

    t0 = time.time()
    if gaussFlag:

        #TODO Ask about the limits on y0
        for i in range(len(dataY)):
            m.limits["y0"+str(i)] = (0, np.inf)

        m.simplex()
        m.migrad() 

    else:

        x = dataX[0]
        def constr(*pars):
            """Constraint for positivity of Gram Carlier.
            Input: All parameters defined in original function.
            Format *pars to work with Minuit.
            x is defined outside function.
            Builds array with all constraints from individual functions."""

            sharedPars = pars[:3]    # sigma1, c4, c6
            joinedGC = np.zeros(int((len(pars)-3)/2) * x.size)
            for i, (A, x0) in enumerate(zip(pars[3::2], pars[4::2])):
                joinedGC[i*x.size : (i+1)*x.size] = model(x, *sharedPars, A, x0)
            
            # if np.any(joinedGC==0):
            #     raise ValueError("Args where zero: ", np.argwhere(joinedGC==0))
            
            return joinedGC

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))
    
    m.hesse()
    t1 = time.time()
    print(f"\nTime of fitting: {t1-t0:.2f} seconds")
    print(f"Value of minimum: {m.fval:.2f}")
    print("\nResults of Global Fit:\n")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p:7s} = {v:7.3f} +/- {e:7.3f}")
    print("\n")

    if showPlots:
        axs = plotData(dataX, dataY, dataE)
        plotFit(dataX, totCost, m, axs)
    return np.array(m.values), np.array(m.errors)


def extractData(ws, wsRes, InstrParsPath, firstSpec, lastSpec):
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    dataRes = wsRes.extractY()
    instrPars = loadInstrParsFileIntoArray(InstrParsPath, firstSpec, lastSpec)
    return dataX, dataY, dataE, dataRes, instrPars    


def loadInstrParsFileIntoArray(InstrParsPath, firstSpec, lastSpec):
    data = np.loadtxt(InstrParsPath, dtype=str)[1:].astype(float)
    spectra = data[:, 0]
    select_rows = np.where((spectra >= firstSpec) & (spectra <= lastSpec))
    instrPars = data[select_rows]
    return instrPars


def takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars):
    zerosRowMask = np.all(dataY==0, axis=1)
    dataY = dataY[~zerosRowMask]
    dataE = dataE[~zerosRowMask]
    dataX = dataX[~zerosRowMask]
    dataRes = dataRes[~zerosRowMask]
    instrPars = instrPars[~zerosRowMask]
    return dataX, dataY, dataE, dataRes, instrPars 


def selectModelAndPars(gaussFlag):
    if gaussFlag:
        def model(x, sigma, y0, A, x0):
            gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
            return gauss 

        defaultPars = {
            "sigma" : 5,
            "y0" : 0,
            "A" : 1,
            "x0" : 0,         
        }

        sharedPars = ["sigma"]

    else:
        def model(x, sigma1, c4, c6, A, x0):
            return A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
        
        defaultPars = {
            "sigma1" : 6,
            "c4" : 0,
            "c6" : 0,
            "A" : 1,
            "x0" : 0          
        }

        sharedPars = ["sigma1", "c4", "c6"]  

    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."

    return model, defaultPars, sharedPars


def calcCostFun(model, i, x, y, yerr, res, sharedPars):
    "Returns cost function for one spectrum i to be summed to total cost function"
   
    xDense, xDelta, resDense = chooseXDense(x, res, True)
    def convolvedModel(xrange, *pars):
        """Performs convolution first on high density grid and interpolates to desired x range"""
        convDense = signal.convolve(model(xDense, *pars), resDense, mode="same") * xDelta
        return np.interp(xrange, xDense, convDense)

    costSig = [key if key in sharedPars else key+str(i) for key in describe(model)]
    convolvedModel.func_code = make_func_code(costSig)
    # print(describe(convolvedModel))

    # Data without cut-offs
    nonZeros = y != 0
    xNZ = x[nonZeros]
    yNZ = y[nonZeros]
    yerrNZ = yerr[nonZeros]

    costFun = cost.LeastSquares(xNZ, yNZ, yerrNZ, convolvedModel)
    return costFun


def chooseXDense(x, res, flag):
    """Make high density symmetric grid for convolution"""

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if flag:
        if x.size % 2 == 0:
            dens = x.size+1  # If even change to odd
        else:
            dens = x.size    # If odd, keep being odd)
    else:
        dens = 1000

    xDense = np.linspace(np.min(x), np.max(x), dens)
    xDelta = xDense[1] - xDense[0]
    resDense = np.interp(xDense, x, res)
    return xDense, xDelta, resDense


def plotData(dataX, dataY, dataE):
    mask = np.all(dataY==0, axis=1)
    dataY = dataY[~mask]
    dataE = dataE[~mask]
    dataX = dataX[~mask]

    rows = 2

    fig, axs = plt.subplots(rows, int(np.ceil(len(dataY)/rows)), 
        figsize=(15, 8), 
        tight_layout=True
        )

    for x, y, yerr, ax in zip(dataX, dataY, dataE, axs.flat):
        ax.errorbar(x, y, yerr, fmt="k.", label="Data")
    return axs


def plotFit(dataX, totCost, minuit, axs):
    for x, costFun, ax in zip(dataX, totCost, axs.flat):
        signature = describe(costFun)

        values = minuit.values[signature]
        errors = minuit.errors[signature]

        yfit = costFun.model(x, *values)

        # Build a decent legend
        leg = []
        for p, v, e in zip(signature, values, errors):
            leg.append(f"${p} = {v:.3f} \pm {e:.3f}$")

        ax.fill_between(x, yfit, label="\n".join(leg), alpha=0.4)
        ax.legend()

# ------- Groupings 

def groupDetectors(ipData, nGroups, showPlots):
    """
    Uses the method of k-means to find clusters in theta-L1 space.
    Input: instrument parameters to extract L1 and theta of detectors.
    Output: list of group lists containing the idx of spectra.
    """
    assert nGroups > 0, "Number of groups must be bigger than zero."
    assert nGroups < len(ipData), "Number of groups cannot exceed no of detectors"
     
    L1 = ipData[:, -1]    
    theta = ipData[:, 2]  

    # Normalize  ranges to similar values
    L1 /= np.sum(L1)       
    theta /= np.sum(theta)

    L1 *= 2           # Bigger weight to L1


    points = np.vstack((L1, theta)).T
    assert points.shape == (len(L1), 2), "Wrong shape."
    centers = points[np.linspace(0, len(points)-1, nGroups).astype(int), :]


    if showPlots:
        plt.scatter(L1, theta, alpha=0.3, color="r", label="Detectors")
        plt.scatter(centers[:, 0], centers[:, 1], color="k", label="Starting centroids")
        plt.xlabel("L1")
        plt.ylabel("theta")
        plt.legend()
        plt.show()

    clusters, n = kMeansClustering(points, centers)
    idxList = formIdxList(clusters, n, len(L1))

    if showPlots:
        for i in range(n):
            clus = points[clusters==i]
            plt.scatter(clus[:, 0], clus[:, 1], label=f"group {i}")
        plt.xlabel("L1")
        plt.ylabel("theta")
        plt.legend()
        plt.show()

    return idxList


def kMeansClustering(points, centers):
    # Fails in some rare situations
    prevCenters = centers
    while  True:
        clusters, nGroups = closestCenter(points, prevCenters)
        centers = calculateCenters(points, clusters, nGroups)
        # print(centers)

        if np.all(centers == prevCenters):
            break

        assert np.isfinite(centers).all(), f"Invalid centers found:\n{centers}\nMaybe try a different number for the groupings."

        prevCenters = centers
    clusters, n = closestCenter(points, centers)
    return clusters, n


def closestCenter(points, centers):
    clusters = np.zeros(len(points))
    for p in range(len(points)):

        minCenter = 0
        minDist = pairDistance(points[p], centers[0])
        for i in range(1, len(centers)): 

            dist = pairDistance(points[p], centers[i])

            if dist < minDist:
                minDist = dist
                minCenter = i
        clusters[p] = minCenter
    return clusters, len(centers)


def pairDistance(p1, p2):
    "pairs have shape (1, 2)"
    return np.sqrt(np.sum(np.square(p1-p2)))


def calculateCenters(points, clusters, n):
    centers = np.zeros((n, 2))
    for i in range(n):
        centers[i] = np.mean(points[clusters==i, :], axis=0)  # If cluster i is not present, returns nan
    return centers


def formIdxList(clusters, n, lenPoints):
    # Form list with groups of idxs
    idxList = []
    for i in range(n):
        idxs = np.argwhere(clusters==i).flatten()
        idxList.append(list(idxs))
    print("\nList of idexes that will be used for idexing: \n", idxList)

    # Check that idexes are not repeated and not missing
    flatList = []
    for group in idxList:
        for elem in group:
            flatList.append(elem)
    assert np.all(np.sort(np.array(flatList))==np.arange(lenPoints)), "Groupings did not work!"
    
    return idxList

# ---------- Weighted Avgs of Groups

def avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList):
    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(dataY[0])))

    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)
        
        if len(groupY) == 1:   # Cannot use weight avg in single spec, wrong results
            meanY, meanE = groupY, groupE
            meanRes = groupRes

        else:
            meanY, meanE = weightedAvgArr(groupY, groupE)
            meanRes = np.nanmean(groupRes, axis=0)

        assert np.all(groupX[0] == np.mean(groupX, axis=0)), "X values should not change with groups"
        
        wDataX[i] = groupX[0]
        wDataY[i] = meanY
        wDataE[i] = meanE
        wDataRes[i] = meanRes 
    
    assert np.all(wDataY!=0), "Some avg weights in groups are not being performed."

    return wDataX, wDataY, wDataE, wDataRes


def initiateZeroArr(shape):
    wDataX = np.zeros(shape)
    wDataY = np.zeros(shape)
    wDataE = np.zeros(shape)
    wDataRes = np.zeros(shape)  
    return  wDataX, wDataY, wDataE, wDataRes


def extractArrByIdx(dataX, dataY, dataE, dataRes, idxs):
    groupE = dataE[idxs, :]
    groupY = dataY[idxs, :]
    groupX = dataX[idxs, :]
    groupRes = dataRes[idxs, :]
    return groupX, groupY, groupE, groupRes


def weightedAvgArr(dataY, dataE):
    meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))
    return meanY, meanE



if __name__ == "__main__":
    main(45, 4, True)    # error at 45 - 24