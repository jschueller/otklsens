import openturns as ot
import math as m

class MetaModelDistributionFactory:
    def __init__(self):
        pass
    def build(self, sample):
        return ot.MetaModelAlgorithm.BuildDistribution(sample)

class KLCoefficientsDistributionFactory:
    def __init__(self):
        pass
    def build(self, sample):

        # try Gaussian with fallback to histogram
        dimension = sample.getDimension()
        marginals = [None] * dimension
        for i in range(dimension):
            sample_i = sample.getMarginal(i)
            level = 0.05 # ResourceMap
            testResult = ot.NormalityTest.CramerVonMisesNormal(sample_i, level)
            if testResult.getBinaryQualityMeasure():
                factory = ot.NormalFactory()
            else:
                factory = ot.HistogramFactory()
            marginals[i] = factory.build(sample_i)
        distribution = ot.ComposedDistribution(marginals)

        # test independence with fallback to beta copula
        isIndependent = True
        for j in range(dimension):
            marginalJ = sample.getMarginal(j)
            for i in range(j + 1, dimension):
                level = 0.05 # ResourceMap
                testResult = ot.HypothesisTest.Spearman(sample.getMarginal(i), marginalJ, level)
                isIndependent = isIndependent and testResult.getBinaryQualityMeasure()
        if not isIndependent:
            betaCopula = ot.EmpiricalBernsteinCopula(sample, sample.getSize())
            distribution.setCopula(betaCopula)

        return distribution
