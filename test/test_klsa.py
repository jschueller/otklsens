import openturns as ot
import openturns.testing as ott
from otklsens import *
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


def test_klcoefdf():
    N = 1000

    x = ot.Normal(3).getSample(N)
    dist = FieldToPointFunctionalChaosAlgorithm.BuildDistribution(x)
    assert 'Normal' in repr(dist.getMarginal(0))
    assert 'IndependentCopula' in repr(dist.getCopula())
    #assert dist.getMarginal(0).getImplementation().__class__.__name__ == 'Normal'
    #assert dist.getCopula().getImplementation().__class__.__name__ == 'IndependentCopula'

    x = ot.Normal([0.0] * 2, ot.CovarianceMatrix([[1.0, 0.8], [0.8, 1.0]])).getSample(N)
    dist = FieldToPointFunctionalChaosAlgorithm.BuildDistribution(x)
    assert 'Normal' in repr(dist.getMarginal(0))
    assert 'EmpiricalBernsteinCopula' in repr(dist.getCopula())
    #assert dist.getMarginal(0).getImplementation().__class__.__name__ == 'Normal'
    #assert dist.getCopula().getImplementation().__class__.__name__ == 'EmpiricalBernsteinCopula'

    x = ot.ComposedDistribution([ot.TruncatedNormal(0.0,1.0,-2.0,2.0)] * 2).getSample(N)
    dist = FieldToPointFunctionalChaosAlgorithm.BuildDistribution(x)
    assert 'Histogram' in repr(dist.getMarginal(0))
    assert 'IndependentCopula' in repr(dist.getCopula())
    #assert dist.getMarginal(0).getImplementation().__class__.__name__ == 'Histogram'
    #assert dist.getCopula().getImplementation().__class__.__name__ == 'IndependentCopula'
    
    x = ot.ComposedDistribution([ot.Uniform()] * 2, ot.GumbelCopula()).getSample(N)
    dist = FieldToPointFunctionalChaosAlgorithm.BuildDistribution(x)
    assert 'Histogram' in repr(dist.getMarginal(0))
    assert 'EmpiricalBernsteinCopula' in repr(dist.getCopula())
    #assert dist.getMarginal(0).getImplementation().__class__.__name__ == 'Histogram'
    #assert dist.getCopula().getImplementation().__class__.__name__ == 'EmpiricalBernsteinCopula'


class pyf2p(ot.OpenTURNSPythonFieldToPointFunction):

    def __init__(self, mesh):
        super(pyf2p, self).__init__(mesh, 3, 2)
        self.setInputDescription(['wn0', 'gp0', 'gp1'])
        self.setOutputDescription(['mean0', 'sumgp'])

    def _exec(self, X):
        Xs = ot.Sample(X)
        mean0 = Xs.computeMean()[0]
        sumgp = Xs[0, 1] + Xs[0, 2]
        Y = [mean0, sumgp]
        return Y

@pytest.fixture
def process_data():
    # input processs
    mesh = ot.RegularGrid(0.0, 0.1, 15)
    cor = ot.CorrelationMatrix(2)
    cor[0, 1] = 0.8
    cov = ot.ExponentialModel([1.0], [1.0, 2.0], cor)
    p1 = ot.WhiteNoise(ot.Normal(0.0, 1.0), mesh)
    p2 = ot.GaussianProcess(cov, mesh)
    process = ot.AggregatedProcess([p1, p2])
    process.setTimeGrid(mesh)
    f = ot.FieldToPointFunction(pyf2p(mesh))
    N = 1000
    x = process.getSample(N)
    y = f(x)
    return x, y

#@pytest.mark.skip
def test_klfce(process_data):
    x, y = process_data
    t0 = time()
    # Verbosity
    ot.Log.Show(ot.Log.INFO)
    #Log.Show(Log.NONE)
    degree = 2
    basisSize = x.getSize()
    algo = FieldToPointFunctionalChaosAlgorithm(x, y)
    algo.setBasisSize(basisSize)
    algo.setThreshold(5e-2)
    algo.setRecompress(True)
    algo.run()
    result = algo.getResult()
    metamodel = result.getFieldToPointMetamodel()
    residuals = result.getResiduals()
    print('residuals=', residuals)
    #assert ot.Point(residuals).norm() < 1e-3, "residual too big"
    sensitivity = FieldFunctionalChaosSobolIndices(result)
    for j in range(y.getDimension()):
        for i in range(x.getDimension()):
            print(f"index({i}, {j}) = {sensitivity.getSobolIndex(i, j)}")
    sobol_0 = [sensitivity.getSobolIndex(i, 0) for i in range(x.getDimension())]
    #ott.assert_almost_equal(sobol_0, [0.366848, 0.428892, 0.201355], 5e-2, 5e-2)

    # now with block indices
    blockIndices = [[0], [1, 2]]
    #dimension = 16
    #basisSize = m.comb(dimension + degree, dimension)
    #print('basisSize=', basisSize)
    algo = FieldToPointFunctionalChaosAlgorithm(x, y)
    algo.setThreshold(5e-2)
    algo.setBasisSize(basisSize)
    algo.setRecompress(False)
    algo.setBlockIndices(blockIndices)
    algo.run()
    result = algo.getResult()
    metamodel = result.getFieldToPointMetamodel()
    residuals = result.getResiduals()
    print('residuals=', residuals)
    #assert ot.Point(residuals).norm() < 1e-3, "residual too big"
    sensitivity = FieldFunctionalChaosSobolIndices(result)
    for j in range(y.getDimension()):
        for i in range(len(blockIndices)):
            print(f"index({i}, {j}) = {sensitivity.getSobolIndex(i, j)}")
    sobol_0 = [sensitivity.getSobolIndex(i, 0) for i in range(len(blockIndices))]

    footer(t0)
