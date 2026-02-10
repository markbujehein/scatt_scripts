from pathlib import Path
from xml.dom import NotFoundErr

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from vesuvio_analysis.core_functions.analysis_functions import (
    calculateMeansAndStds,
    filterWidthsAndIntensities,
)
from vesuvio_analysis.core_functions.fit_in_yspace import selectModelAndPars
from vesuvio_analysis.core_functions.ICHelpers import setBootstrapDirs

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FIXED: Use Current Working Directory (CWD) instead of library location
# This ensures outputs go to your project folder, not the library source
## OLD CODE:
# currentPath = Path(__file__).parent.absolute()
# experimentsPath = currentPath / ".." / ".. " / "experiments"
# IPFilesPath = currentPath / ".." / "ip_files"
## NEW CODE:
experimentsPath = Path.cwd() / "experiments"
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr):
    if not (analysisIC.runAnalysis):
        return

    setBootstrapDirs(
        bckwdIC, fwdIC, bootIC, yFitIC
    )  # Same function used to store data, to check below if dirs exist

    for IC in [bckwdIC, fwdIC]:
        if not (IC.bootSavePath.is_file()):
            print("Bootstrap data files not found, unable to run analysis!")
            print(f"{IC.bootSavePath.name}")
            continue  # If main results are not present, assume ysapce results are also missing

        checkLogMatch(IC, isYFitFile=False)

        bootParsRaw, parentParsRaw, nSamples, corrResiduals = readBootData(
            IC.bootSavePath
        )
        checkResiduals(corrResiduals)
        checkBootSamplesVSParent(bootParsRaw, parentParsRaw, IC)  # Prints comparison

        bootPars = (
            bootParsRaw.copy()
        )  # By default do not filter means, copy to avoid accidental changes
        if analysisIC.filterAvg:
            bootPars = filteredBootMeans(bootParsRaw.copy(), IC)
            try:
                print("\nCompare filtered parameters with parent:\n")
                checkBootSamplesVSParent(
                    bootPars, parentParsRaw, IC
                )  # Prints comparison
            except AssertionError:
                print("\nUnable to calculate new means of filtered parameters.\n")

        # Plots histograms of all spectra for a given width or intensity
        plotRawWidthsAndIntensities(analysisIC, IC, bootPars, parentParsRaw)

        # Calculate bootstrap histograms for mean widths and intensities
        meanWidths, meanIntensities = calculateMeanWidthsIntensities(
            bootPars, IC, nSamples
        )

        # If filer is on, check that it matches original procedure
        checkMeansProcedure(analysisIC, IC, meanWidths, meanIntensities, bootParsRaw)

        plotMeanWidthsAndIntensities(
            analysisIC, IC, meanWidths, meanIntensities, parentParsRaw
        )
        plotMeansEvolution(analysisIC, meanWidths, meanIntensities)
        plot2DHistsWidthsAndIntensities(analysisIC, meanWidths, meanIntensities)

        if not (IC.bootYFitSavePath.is_file()):
            print(
                "Bootstrap data file for y-space fit not found, unable to run analysis!"
            )
            print(f"{IC.bootYFitSavePath.name}")
            continue

        checkLogMatch(IC, isYFitFile=True)

        minuitFitVals = readYFitData(IC.bootYFitSavePath, yFitIC)

        plotMeansEvolutionYFit(analysisIC, minuitFitVals)
        plotYFitHists(analysisIC, yFitIC, minuitFitVals)
        plot2DHistsYFit(analysisIC, minuitFitVals)


def checkLogMatch(IC, isYFitFile):
    """Checks if currently selected data file matches stored logs."""
    currentLog = IC.bootYFitSavePathLog if isYFitFile else IC.bootSavePathLog
    currName = currentLog.split(" : ")[0]
    # Check sample present in log file
    with open(IC.logFilePath, "r") as logFile:
        for line in logFile:
            # Check if name of file is present
            if line.split(" : ")[0] == currName:
                if line.strip("\n") == currentLog:
                    return
                raise NotFoundErr(
                    currName + " found but corresponding log does not match."
                )
        raise NotFoundErr(currName + " not found in logs file")


def readBootData(dataPath):
    bootData = np.load(dataPath)

    # Select fitting parameters of ncp
    # Discard first column of spectrum number
    # Discard last two columns of number of iterations and chi2
    bootParsRaw = bootData["boot_samples"][:, :, 1:-2]
    parentParsRaw = bootData["parent_result"][:, 1:-2]

    nSamples = len(bootParsRaw)
    try:
        corrResiduals = bootData["corr_residuals"]
    except KeyError:
        corrResiduals = np.array([np.nan])
        print("\nCorrelation of coefficients not found!\n")

    failMask = np.all(np.isnan(bootParsRaw), axis=(1, 2))
    assert failMask.shape == (len(bootParsRaw),), (
        f"Wrong shape of masking: {failMask.shape} != {bootParsRaw.shape} "
    )
    if np.sum(failMask) > 0:
        print(f"\nNo of failed samples: {np.sum(failMask)}")
        print("\nUsing only good replicas ...\n")
        bootParsRaw = bootParsRaw[~failMask]
        nSamples = np.sum(~failMask)

    maskedIdxs = np.where(np.all(np.isnan(bootParsRaw), axis=(0, 2)))
    print(f"Masked idxs with nans found: {maskedIdxs}")
    print(f"\nData files found:\n{dataPath.name}")
    print(f"\nNumber of samples in the file: {nSamples}")
    assert ~np.all(bootParsRaw[-1] == parentParsRaw), (
        "Error in Jackknife due to last column."
    )
    return bootParsRaw, parentParsRaw, nSamples, corrResiduals


def readYFitData(dataPath, yFitIC):
    """Resulting output has shape(no of pars, no of samples)"""

    fitIdx = 0  # 0 to select Minuit, 1 to select LM

    bootYFitData = np.load(dataPath)
    try:
        bootYFitVals = bootYFitData["boot_samples"]
    except KeyError:
        bootYFitVals = bootYFitData["boot_vals"]  # To account for some previous samples
    minFitVals = bootYFitVals[
        :, fitIdx, :-1
    ].T  # Select Minuit values and Discard last value chi2

    failMask = np.all(np.isnan(minFitVals), axis=0)
    if np.sum(failMask) > 0:
        print(f"\nNo of failed samples: {np.sum(failMask)}")
        print("\nUsing only good replicas ...\n")
        minFitVals = minFitVals[:, ~failMask]

    try:
        parentPopt = bootYFitData["parent_popt"][fitIdx]
        parentPerr = bootYFitData["parent_perr"][fitIdx]
        printYFitParentPars(yFitIC, parentPopt, parentPerr)  # TODO: Test this function
    except KeyError:
        pass
    return minFitVals


def checkResiduals(corrRes):
    if np.all(np.isnan(corrRes)):
        return

    corrCoef = corrRes[:, 0]
    nCorrelated = np.sum(corrCoef > 0.5)
    print(f"\nNumber of spectra with pearson r > 0.5: {nCorrelated}")
    return


def checkBootSamplesVSParent(bestPars, parentPars, IC):
    """
    For an unbiased estimator, the mean of the bootstrap samples will converge to
    the mean of the experimental sample (here called parent).
    """

    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    meanBootWidths = np.mean(bootWidths, axis=0)
    meanBootIntensities = np.mean(bootIntensities, axis=0)

    avgWidths, stdWidths, avgInt, stdInt = calculateMeansAndStds(
        meanBootWidths.T, meanBootIntensities.T, IC
    )

    parentWidths = parentPars[:, 1::3]
    parentIntensities = parentPars[:, 0::3]

    avgWidthsP, stdWidthsP, avgIntP, stdIntP = calculateMeansAndStds(
        parentWidths.T, parentIntensities.T, IC
    )

    print("\nComparing Bootstrap means with parent means:\n")
    printResults(avgWidths, stdWidths, "Boot Widths")
    printResults(avgWidthsP, stdWidthsP, "Parent Widths")
    printResults(avgInt, stdInt, "Boot Intensities")
    printResults(avgIntP, stdIntP, "Parent Intensities")


def printResults(arrM, arrE, mode):
    print(f"\n{mode}:\n")
    for i, (m, e) in enumerate(zip(arrM, arrE)):
        print(f"{mode} {i}: {m:>6.3f} \u00b1 {e:<6.3f}")


def filteredBootMeans(
    bestPars, IC
):  # Pass IC just to check flag for preliminary procedure
    """Use same filtering function used on original procedure"""

    # Extract Widths and Intensities from bootstrap samples
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    # Perform the filter
    for i, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):
        filteredWidths, filteredIntensities = filterWidthsAndIntensities(
            widths.T, intensities.T, IC
        )

        bootWidths[i] = filteredWidths.T
        bootIntensities[i] = filteredIntensities.T

    # Convert back to format of bootstrap samples
    filteredBestPars = bestPars.copy()
    filteredBestPars[:, :, 1::3] = bootWidths
    filteredBestPars[:, :, 0::3] = bootIntensities
    return filteredBestPars


def plotRawWidthsAndIntensities(analysisIC, IC, bootPars, parentPars):
    """
    Plots histograms of each width and intensity seperatly.
    Plots histogram of means over spectra for each width or intensity.
    """

    if not (analysisIC.plotRawWidthsIntensities):
        return

    parentWidths, parentIntensities = extractParentMeans(parentPars, IC)
    noOfMasses = len(parentWidths)

    fig, axs = plt.subplots(2, noOfMasses)

    for axIdx, startIdx, kind, parentMeans in zip(
        [0, 1], [1, 0], ["Width", "Intensity"], [parentWidths, parentIntensities]
    ):
        for i, j in enumerate(range(startIdx, 3 * noOfMasses, 3)):
            axs[axIdx, i].set_title(f"{kind} {i}")
            idxSamples = selectRawSamplesPerIdx(bootPars, j)
            plotHists(axs[axIdx, i], idxSamples, disableCI=True, disableLeg=True)
            axs[axIdx, i].axvline(
                parentMeans[i], 0.75, 0.97, color="b", ls="-", alpha=0.4
            )

    plt.show()
    return


def extractParentMeans(parentPars, IC):
    """Uses original treatment of widths and intensities to calculate parent means."""
    # Modify so that the filtering doesn't happen by default
    meanWp, meanIp = calcMeansWithOriginalProc(parentPars[np.newaxis, :, :], IC)
    meanWp = meanWp.flatten()
    meanIp = meanIp.flatten()
    return meanWp, meanIp


def calcMeansWithOriginalProc(bestPars, IC):
    """Performs the means and std on each bootstrap sample according to original procedure"""

    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    bootMeanW = np.zeros((len(bootWidths[0, 0, :]), len(bootWidths)))
    bootStdW = np.zeros(bootMeanW.shape)
    bootMeanI = np.zeros(bootMeanW.shape)
    bootStdI = np.zeros(bootMeanW.shape)

    for j, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):
        meanW, stdW, meanI, stdI = calculateMeansAndStds(widths.T, intensities.T, IC)

        bootMeanW[:, j] = meanW  # Interested only in the means
        bootMeanI[:, j] = meanI

    return bootMeanW, bootMeanI


def calculateMeanWidthsIntensities(bootPars, IC, nSamples):
    """
    Calculates means for each Bootstrap sample.
    Returns means with size equal to the number of Boot samples.
    """

    # Calculate bootstrap histograms for mean widths and intensities
    meanWidths = np.zeros((len(IC.masses), nSamples))
    for i, j in enumerate(range(1, 3 * len(IC.masses), 3)):
        idxSamples = selectRawSamplesPerIdx(bootPars, j)
        meanWidths[i, :] = np.nanmean(idxSamples, axis=0)

    meanIntensities = np.zeros((len(IC.masses), nSamples))
    for i, j in enumerate(range(0, 3 * len(IC.masses), 3)):
        idxSamples = selectRawSamplesPerIdx(bootPars, j)
        meanIntensities[i, :] = np.nanmean(idxSamples, axis=0)

    return meanWidths, meanIntensities


def checkMeansProcedure(analysisIC, IC, meanWidths, meanIntensities, bootParsRaw):
    """Checks that filtering and averaging of Bootstrap samples follows the original procedure"""

    if not (
        analysisIC.filterAvg
    ):  # When filtering not present, no comparison available
        return

    else:  # Check that treatment of data matches original
        meanWOri, meanIOri = calcMeansWithOriginalProc(bootParsRaw, IC)
        np.testing.assert_array_almost_equal(meanWOri, meanWidths)
        np.testing.assert_array_almost_equal(meanIOri, meanIntensities)
        return


def plotMeanWidthsAndIntensities(
    analysisIC, IC, meanWidths, meanIntensities, parentParsRaw
):
    """
    Most informative histograms, shows all mean widhts and intensities of Bootstrap samples
    """

    if not (analysisIC.plotMeanWidthsIntensities):
        return

    parentWidths, parentIntensities = extractParentMeans(parentParsRaw, IC)

    print("\n\n Test passed! Mean Widths match!")

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Histograms of mean Widths")
    axs[1].set_title("Histograms of mean Intensitiess")

    for ax, means, parentMeans in zip(
        axs.flatten(), [meanWidths, meanIntensities], [parentWidths, parentIntensities]
    ):
        plotHists(ax, means, disableAvg=True, disableCI=True)
        for pMean in parentMeans:
            ax.axvline(pMean, 0.75, 0.97, color="b", ls="-", alpha=0.4)

    plt.show()
    return


def plotMeansEvolution(IC, meanWidths, meanIntensities):
    """Shows how results of Bootstrap change depending on number of bootstrap samples"""

    if not (IC.plotMeansEvolution):
        return

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title("Evolution of mean Widths")
    plotMeansOverNoSamples(axs[0, 0], meanWidths)

    axs[1, 0].set_title("Widths normalized to last value")
    plotMeansOverNoSamples(axs[1, 0], meanWidths, normFlag=True)

    axs[0, 1].set_title("Evolution of mean Intensities")
    plotMeansOverNoSamples(axs[0, 1], meanIntensities)

    axs[1, 1].set_title("Intensities normalized to last value")
    plotMeansOverNoSamples(axs[1, 1], meanIntensities, normFlag=True)

    plt.show()
    return


def plotMeansEvolutionYFit(analysisIC, minuitFitVals):
    if not (analysisIC.plotMeansEvolution):
        return

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("Evolution of y-space fit parameters")
    plotMeansOverNoSamples(ax[0], minuitFitVals)

    ax[1].set_title("y-space fit parameters normalized to last value")
    plotMeansOverNoSamples(ax[1], minuitFitVals, normFlag=True)

    plt.show()
    return


def plotMeansOverNoSamples(ax, bootMeans, normFlag=False):
    nSamples = len(bootMeans[0])
    assert nSamples >= 10, "To plot evolution of means, need at least 10 samples!"
    noOfPoints = int(nSamples / 10)
    sampleSizes = np.linspace(10, nSamples, noOfPoints).astype(int)

    sampleMeans = np.zeros((len(bootMeans), len(sampleSizes)))
    sampleBounds = np.zeros((len(bootMeans), 2, len(sampleSizes)))

    for i, N in enumerate(sampleSizes):
        subSample = bootMeans[:, :N].copy()

        mean = np.mean(subSample, axis=1)

        bounds = np.percentile(subSample, [16, 68 + 16], axis=1).T  # 1 standard dev
        assert bounds.shape == (len(subSample), 2), f"Wrong shape: {bounds.shape}"
        # errors = bounds - mean[:, np.newaxis]

        sampleMeans[:, i] = mean
        sampleBounds[:, :, i] = bounds

    meansFinal = sampleMeans
    boundsFinal = sampleBounds
    ylabel = "Means and Errors Values"
    if (
        normFlag
    ):  # Rescale and normalize to last value, so all points are converging to zero
        lastValue = sampleMeans[:, -1][:, np.newaxis]
        meansFinal = (
            (sampleMeans - lastValue) / lastValue * 100
        )  # Percent change to last value
        boundsFinal = (
            (sampleBounds - lastValue[:, np.newaxis, :])
            / lastValue[:, np.newaxis, :]
            * 100
        )
        ylabel = "Percent Change [%]"

    for i, (means, errors) in enumerate(zip(meansFinal, boundsFinal)):
        ax.plot(sampleSizes, means, label=f"idx {i}")
        ax.fill_between(sampleSizes, errors[0, :], errors[1, :], alpha=0.2)

    ax.legend()
    ax.set_xlabel("Number of Bootstrap samples")
    ax.set_ylabel(ylabel)


def plot2DHistsWidthsAndIntensities(IC, meanWidths, meanIntensities):
    if not (IC.plot2DHists):
        return

    assert meanWidths.shape == meanIntensities.shape, (
        "Widths and Intensities need to be the same shape."
    )

    plot2DHists(meanWidths, "Widths")
    plot2DHists(meanIntensities, "Intensities")
    return


def plot2DHists(bootSamples, mode):
    """bootSamples has histogram rows for each parameter"""

    plotSize = len(bootSamples)
    fig, axs = plt.subplots(plotSize, plotSize, tight_layout=True)

    for i in range(plotSize):
        for j in range(plotSize):
            # axs[i, j].axis("off")

            if j > i:
                axs[i, j].set_visible(False)

            elif i == j:
                if i > 0:
                    orientation = "horizontal"
                else:
                    orientation = "vertical"

                axs[i, j].hist(bootSamples[i], orientation=orientation)

            else:
                axs[i, j].hist2d(bootSamples[j], bootSamples[i])

            if j == 0:
                axs[i, j].set_ylabel(f"idx {i}")
            else:
                axs[i, j].get_yaxis().set_ticks([])

            if i == plotSize - 1:
                axs[i, j].set_xlabel(f"idx{j}")
            else:
                axs[i, j].get_xaxis().set_ticks([])

            axs[i, j].set_title(
                f"r = {stats.pearsonr(bootSamples[i], bootSamples[j])[0]:.3f}"
            )

    fig.suptitle(f"1D and 2D Histograms of {mode}")

    plt.show()


def printYFitParentPars(yFitIC, parentPopt, parentPerr):
    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)
    sig = [p for p in defaultPars]
    sig = ["y0"] + sig  # Add intercept from outside function parameters

    print("\nParent parameters of y-sapce fit:\n")
    for p, m, e in zip(sig, parentPopt, parentPerr):
        print(f"{p:5s}:  {m:8.3f} +/- {e:8.3f}")


def plotYFitHists(analysisIC, yFitIC, yFitHists):
    """Histogram for each parameter of model used for fit in y-space."""

    if not (analysisIC.plotYFitHists):
        return

    # Plot each parameter in an individual histogram
    fig, axs = plt.subplots(
        2, int(np.ceil(len(yFitHists) / 2)), figsize=(12, 7), tight_layout=True
    )

    # To label each histogram, extract signature of function used for the fit
    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)
    sig = [
        p for p in defaultPars
    ]  # Assumes defaultPars in same order as signature of the function
    sig = ["y0"] + sig  # Add intercept from outside function parameters

    for i, (ax, hist, par) in enumerate(zip(axs.flatten(), yFitHists, sig)):
        ax.set_title(f"Fit Parameter: {par}")
        plotHists(ax, hist[np.newaxis, :], disableAvg=True)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:  # If empty list
            ax.set_visible(False)

    plt.show()
    return


def plot2DHistsYFit(analysisIC, minuitFitVals):
    if not (analysisIC.plot2DHists):
        return

    plot2DHists(minuitFitVals, "Y-space Fit parameters")
    return


def plotHists(ax, samples, disableCI=False, disableLeg=False, disableAvg=False):
    """Plots each row of 2D samples array as a histogram."""

    # ax.set_title(f"Histogram of {title}")
    for i, bootHist in enumerate(samples):
        if np.all(bootHist == 0) or np.all(np.isnan(bootHist)):
            continue

        mean = np.nanmean(bootHist)
        bounds = np.percentile(bootHist, [16, 68 + 16])  # 1 std: 68%, 2 std: 95%
        errors = bounds - mean
        leg = f"Row {i}: {mean:>6.3f} +{errors[1]:.3f} {errors[0]:.3f}"

        ax.hist(bootHist, histtype="step", label=leg, linewidth=1)
        ax.axvline(mean, 0.9, 0.97, color="k", ls="--", alpha=0.4)

        if not (disableCI):
            ax.axvspan(bounds[0], bounds[1], alpha=0.1, color="b")

    if not (disableAvg):
        # Plot average over histograms
        avgHist = np.nanmean(samples, axis=0).flatten()
        ax.hist(avgHist, color="r", histtype="step", linewidth=2)
        ax.axvline(np.nanmean(avgHist), 0.75, 0.97, color="r", ls="--", linewidth=2)

    if not (disableLeg):
        ax.legend(loc="upper center")


def selectRawSamplesPerIdx(bootSamples, idx):
    """
    Returns samples and mean for a single width or intensity at a given index.
    Samples shape: (No of spectra, No of boot samples)
    Samples mean: Mean of spectra (mean of histograms)
    """
    samples = bootSamples[:, :, idx].T
    return samples


# TODO: Make use of this function?
def calcCorrWithScatAngle(samples, IC):
    """Calculate correlation coefficient between histogram means and scattering angle."""

    ipMatrix = np.loadtxt(IC.instrParsPath, dtype=str)[1:].astype(float)

    firstSpec, lastSpec = IC.bootSavePath.name.split("_")[1].split("-")
    allSpec = ipMatrix[:, 0]
    selectedSpec = (allSpec >= int(firstSpec)) & (allSpec >= int(lastSpec))

    thetas = ipMatrix[selectedSpec, 2]  # Scattering angle on third column

    # thetas = ipMatrix[firstIdx : lastIdx, 2]    # Scattering angle on third column
    assert thetas.shape == (len(samples),), f"Wrong shape: {thetas.shape}"

    histMeans = np.nanmean(samples, axis=1)

    # Remove masked spectra:
    nanMask = np.isnan(histMeans)
    histMeans = histMeans[~nanMask]
    thetas = thetas[~nanMask]

    corr = stats.pearsonr(thetas, histMeans)
    return corr
