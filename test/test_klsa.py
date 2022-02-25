import openturns as ot
from otklsens import KarhunenLoeveSensitivityAnalysis
import math as m
from time import time
from openturns.viewer import View

def header(msg):
    t0 = time()
    print("="*len(msg))
    print(msg)
    print("="*len(msg))
    return(t0)

def footer(t0):
    print("t=", time() - t0, "s")

def test_klfce():
    t00 = time()
    # Verbosity
    ot.Log.Show(ot.Log.INFO)
    #Log.Show(Log.NONE)

    
    footer(t0)
    
