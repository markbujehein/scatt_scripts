# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def convert_symetrise_yspace_ori(ws_name,mass):
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
    Rebin(InputWorkspace=ws_name+"_JoY", Params = rebin_parameters,FullBinsOnly=True, OutputWorkspace= ws_name+"_JoY")
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    for j in range(tmp.getNumberHistograms()):
        for k in range(tmp.blocksize()):
            tmp.dataE(j)[k] =0.
            if (tmp.dataY(j)[k]!=0):
                tmp.dataY(j)[k] =1.
    tmp=SumSpectra('tmp')
    SumSpectra(InputWorkspace=ws_name+"_JoY",OutputWorkspace=ws_name+"_JoY")
    Divide(LHSWorkspace=ws_name+"_JoY", RHSWorkspace="tmp", OutputWorkspace =ws_name+"_JoY")
    ws=mtd[ws_name+"_JoY"]
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    for k in range(tmp.blocksize()):
        tmp.dataE(0)[k] =(ws.dataE(0)[k]+ws.dataE(0)[ws.blocksize()-1-k])/2.
        tmp.dataY(0)[k] =(ws.dataY(0)[k]+ws.dataY(0)[ws.blocksize()-1-k])/2
    RenameWorkspace(InputWorkspace="tmp",OutputWorkspace=ws_name+"_JoYori")
    normalise_workspace(ws_name+"_JoYori")      #originally _JoY
    return max_Y, mtd[ws_name+"_JoYori"]
 
def convert_symetrise_yspace_opt(ws_name, mass):  
    """input: TOF workspace
       output: workspace in y-space for given mass with dataY symetrised"""
          
    ws_y, ws_q = ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
    max_Y = np.ceil(2.5*mass+27)    #where from
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)   #first bin boundary, width, last bin boundary, so 120 bins over range
    ws_y = Rebin(InputWorkspace=ws_y, Params = rebin_parameters, FullBinsOnly=True, OutputWorkspace=ws_name+"_JoY")
   
    matrix_Y = ws_y.extractY()        
    matrix_Y[(matrix_Y != 0) & (matrix_Y != np.nan)] = 1       #safeguarding against nans as well
    no_y = np.nansum(matrix_Y, axis=0)   
    
    ws_y = SumSpectra(InputWorkspace=ws_y, OutputWorkspace=ws_name+"_JoY")
    tmp = CloneWorkspace(InputWorkspace=ws_y)
    tmp.dataY(0)[:] = no_y
    tmp.dataE(0)[:] = np.zeros(tmp.blocksize())
    
    ws = Divide(LHSWorkspace=ws_y, RHSWorkspace=tmp, OutputWorkspace =ws_name+"_JoYopt")
    ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2           #symetrise dataY
    ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2           #symetrise dataE
    normalise_workspace(ws)
    return max_Y, ws


def test_func(ws, mass):
    
    y, ori_ws = convert_symetrise_yspace_ori(ws, mass)
    y, opt_ws = convert_symetrise_yspace_opt(ws, mass)
    
    dataY_ori, dataE_ori, dataX_ori = ori_ws.extractY(), ori_ws.extractE(), ori_ws.extractX()
    dataY_opt, dataE_opt, dataX_opt = opt_ws.extractY(), opt_ws.extractE(), opt_ws.extractX()
    
    np.testing.assert_allclose(dataY_ori, dataY_opt)
    np.testing.assert_allclose(dataE_ori, dataE_opt)
    np.testing.assert_allclose(dataX_ori, dataX_opt)
    
    
ws = Load(Filename="")
   
test_func(ws.name(), 1.0079)
    
    
        