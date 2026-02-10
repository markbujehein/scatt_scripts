import numpy as np
from mantid.simpleapi import *


def fitGlobalMantidFit(wsJoY, wsQ, wsRes, minimizer, yFitIC, wsFirstMass):
    """Uses original Mantid procedure to perform global fit on groups with 8 detectors"""

    replaceNansWithZeros(wsJoY)
    wsGlobal = artificialErrorsInUnphysicalBins(wsJoY)
    wsQInv = createOneOverQWs(wsQ)
    avgWidths = mantidGlobalFitProcedure(
        wsGlobal, wsQInv, wsRes, minimizer, yFitIC, wsFirstMass
    )


def replaceNansWithZeros(ws):
    for j in range(ws.getNumberHistograms()):
        ws.dataY(j)[np.isnan(ws.dataY(j)[:])] = 0
        ws.dataE(j)[np.isnan(ws.dataE(j)[:])] = 0


def artificialErrorsInUnphysicalBins(wsJoY):
    wsGlobal = CloneWorkspace(
        InputWorkspace=wsJoY, OutputWorkspace=wsJoY.name() + "_Global"
    )
    for j in range(wsGlobal.getNumberHistograms()):
        wsGlobal.dataE(j)[wsGlobal.dataE(j)[:] == 0] = 0.1

    assert np.any(np.isnan(wsGlobal.extractE())) == False, (
        "Nan present in input workspace need to be replaced by zeros."
    )

    return wsGlobal


def createOneOverQWs(wsQ):
    wsInvQ = CloneWorkspace(InputWorkspace=wsQ, OutputWorkspace=wsQ.name() + "_Inverse")
    for j in range(wsInvQ.getNumberHistograms()):
        nonZeroFlag = wsInvQ.dataY(j)[:] != 0
        wsInvQ.dataY(j)[nonZeroFlag] = 1 / wsInvQ.dataY(j)[nonZeroFlag]

        ZeroIdxs = np.argwhere(wsInvQ.dataY(j)[:] == 0)  # Indxs of zero elements
        if ZeroIdxs.size != 0:  # When zeros are present
            wsInvQ.dataY(j)[ZeroIdxs[0] - 1] = 0  # Put a zero before the first zero

    return wsInvQ


def mantidGlobalFitProcedure(wsGlobal, wsQInv, wsRes, minimizer, yFitIC, wsFirstMass):
    """Original Implementation of Global Fit using Mantid"""

    if yFitIC.fitModel == "SINGLE_GAUSSIAN":
        convolution_template = """
        (composite=Convolution,$domains=({0});
        name=Resolution,Workspace={1},WorkspaceIndex={0};
            (
            name=UserFunction,Formula=
            A*exp( -(x-x0)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5,
            A=1.,x0=0.,Sigma=6.0,  ties=();
                (
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace={2},WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
                name=UserFunction,Formula=
                Sigma*1.4142/12.*exp( -(x)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
                *((8*((x)/sqrt(2.)/Sigma)^3-12*((x)/sqrt(2.)/Sigma))),
                Sigma=6.0);ties=()
                )
            )"""
    elif yFitIC.fitModel == "GC_C4_C6":
        convolution_template = """
        (composite=Convolution,$domains=({0});
        name=Resolution,Workspace={1},WorkspaceIndex={0};
            (
            name=UserFunction,Formula=
            A*exp( -(x-x0)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
            *(1+c4/32*(16*((x-x0)/sqrt(2.)/Sigma)^4-48*((x-x0)/sqrt(2.)/Sigma)^2+12)),
            A=1.,x0=0.,Sigma=6.0, c4=0, ties=();
                (
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace={2},WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
                name=UserFunction,Formula=
                Sigma*1.4142/12.*exp( -(x)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
                *((8*((x)/sqrt(2.)/Sigma)^3-12*((x)/sqrt(2.)/Sigma))),
                Sigma=6.0);ties=()
                )
            )"""
    else:
        raise ValueError(
            f"{yFitIC.fitModel} not supported for Global Mantid fit. Supported options: 'SINGLE_GAUSSIAN', 'GC_C4_C6'"
        )

    print("\nGlobal fit in the West domain over 8 mixed banks\n")
    widths = []
    for bank in range(8):
        dets = [bank, bank + 8, bank + 16, bank + 24]

        convolvedFunctionsList = []
        ties = ["f0.f1.f1.f1.Sigma=f0.f1.f0.Sigma"]
        datasets = {"InputWorkspace": wsGlobal.name(), "WorkspaceIndex": dets[0]}

        print("Detectors: ", dets)

        counter = 0
        for i in dets:
            print(f"Considering spectrum {wsGlobal.getSpectrumNumbers()[i]}")
            if wsGlobal.spectrumInfo().isMasked(i):
                print(f"Skipping masked spectrum {wsGlobal.getSpectrumNumbers()[i]}")
                continue

            thisIterationFunction = convolution_template.format(
                counter, wsRes.name(), wsQInv.name()
            )
            convolvedFunctionsList.append(thisIterationFunction)

            if counter > 0:
                if yFitIC.fitModel == "SINGLE_GAUSSIAN":
                    ties.append(
                        "f{0}.f1.f0.Sigma= f{0}.f1.f1.f1.Sigma=f0.f1.f0.Sigma".format(
                            counter
                        )
                    )
                else:
                    ties.append(
                        "f{0}.f1.f0.Sigma= f{0}.f1.f1.f1.Sigma=f0.f1.f0.Sigma".format(
                            counter
                        )
                    )
                    ties.append("f{0}.f1.f0.c4=f0.f1.f0.c4".format(counter))
                    ties.append("f{0}.f1.f1.f1.c3=f0.f1.f1.f1.c3".format(counter))

                # Attach datasets
                datasets[f"InputWorkspace_{counter}"] = wsGlobal.name()
                datasets[f"WorkspaceIndex_{counter}"] = i
            counter += 1

        multifit_func = f"composite=MultiDomainFunction; {';'.join(convolvedFunctionsList)}; ties=({','.join(ties)})"
        minimizer_string = (
            f"{minimizer}, AbsError=0.00001, RealError=0.00001, MaxIterations=2000"
        )

        # Unpack dictionary as arguments
        Fit(
            multifit_func,
            Minimizer=minimizer_string,
            Output=wsFirstMass.name() + f"_Joy_Mixed_Banks_Bank_{str(bank)}_fit",
            **datasets,
        )

        # Select ws with fit results
        ws = mtd[
            wsFirstMass.name() + f"_Joy_Mixed_Banks_Bank_{str(bank)}_fit_Parameters"
        ]
        print(f"Bank: {str(bank)} -- sigma={ws.cell(2, 1)} +/- {ws.cell(2, 2)}")
        widths.append(ws.cell(2, 1))

        DeleteWorkspace(
            wsFirstMass.name()
            + "_Joy_Mixed_Banks_Bank_"
            + str(bank)
            + "_fit_NormalisedCovarianceMatrix"
        )
        DeleteWorkspace(
            wsFirstMass.name()
            + "_Joy_Mixed_Banks_Bank_"
            + str(bank)
            + "_fit_Workspaces"
        )

    print(
        "\nAverage hydrogen standard deviation: ",
        np.mean(widths),
        " +/- ",
        np.std(widths),
    )
    return widths
