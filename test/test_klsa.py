import openturns as ot
from otklsens import FieldToPointKarhunenLoeveFCEAlgorithm, FieldToPointKarhunenLoeveSensitivityAnalysis
import math as m
from time import time
from openturns.viewer import View
import pytest

def header(msg):
    t0 = time()
    print("="*len(msg))
    print(msg)
    print("="*len(msg))
    return t0

def footer(t0):
    print("t=", time() - t0, "s")


class pyf2p(ot.OpenTURNSPythonFieldToPointFunction):

    def __init__(self, mesh):
        super(pyf2p, self).__init__(mesh, 3, 3)
        self.setInputDescription(['WN0', 'GP0', 'GP1'])
        self.setOutputDescription(['mean0', 'mean1', 'mean2'])

    def _exec(self, X):
        Y = ot.Sample(X).computeMean()
        return Y

@pytest.fixture
def process_data():
    # input processs
    mesh = ot.RegularGrid(0.0, 0.1, 11)
    cor = ot.CorrelationMatrix(2)
    cor[0, 1] = 0.8
    cov = ot.ExponentialModel([1.0], [1.0, 2.0], cor)
    p2 = ot.GaussianProcess(cov, mesh)
    process = ot.AggregatedProcess([ot.WhiteNoise(), p2])
    process.setTimeGrid(mesh)
    f = ot.FieldToPointFunction(pyf2p(mesh))
    N = 1000
    x = process.getSample(N)    
    y = f(x)
    return x, y

def test_klfce(process_data):
    x, y = process_data
    t0 = time()
    # Verbosity
    ot.Log.Show(ot.Log.INFO)
    #Log.Show(Log.NONE)
    algo = FieldToPointKarhunenLoeveFCEAlgorithm(x, y)
    algo.run()
    result = algo.getResult()
    metamodel = result.getMetaModel()
    residuals = result.getResiduals()
    print('residuals=', residuals)
    assert ot.Point(residuals).norm() < 1e-2, "residual too big"
    footer(t0)

def test_klsens(process_data):
    x, y = process_data
    t0 = time()
    # Verbosity
    ot.Log.Show(ot.Log.INFO)
    #Log.Show(Log.NONE)
    algo = FieldToPointKarhunenLoeveSensitivityAnalysis(x, y)
    algo.run()
    result = algo.getResult()
    footer(t0) 
    
