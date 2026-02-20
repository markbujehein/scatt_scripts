from vesuvio_analysis.core_functions.run_script import runScript
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from vesuvio_analysis.tests.tests_IC import scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC
testPath = Path(__file__).absolute().parent 

np.random.seed(3)   # Set seed so that tests match everytime

class BootstrapInitialConditions:
    runBootstrap = True

    procedure = "JOINT"
    fitInYSpace = None

    bootstrapType = "JACKKNIFE"
    nSamples = 3   # Overwritten by running Jackknife
    skipMSIterations = False
    runningTest = True
    userConfirmation = False


class UserScriptControls:
    runRoutine = False 
    procedure = "FORWARD"   
    fitInYSpace = None    
    # bootstrap = "JOINT"   

bootIC = BootstrapInitialConditions
userCtr = UserScriptControls

bootRes, noneRes = runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

jackBackSamples = bootRes["bckwdScat"].bootSamples
jackFrontSamples = bootRes["fwdScat"].bootSamples


# jackJointResults = bootRes

# jackSamples = []
# for jackRes in jackJointResults:
#     jackSamples.append(jackRes.bootSamples)

# jackBackSamples, jackFrontSamples = jackSamples


oriJackBack = testPath / "stored_joint_jack_back.npz"
oriJackFront = testPath / "stored_joint_jack_front.npz"

class TestJointBootstrap(unittest.TestCase):

    def setUp(self):
        self.oriJointBack = np.load(oriJackBack)["boot_samples"]
        self.oriJointFront = np.load(oriJackFront)["boot_samples"]

    def testBack(self):
        nptest.assert_array_almost_equal(jackBackSamples, self.oriJointBack)

    def testFront(self):
        nptest.assert_array_almost_equal(jackFrontSamples, self.oriJointFront)

 