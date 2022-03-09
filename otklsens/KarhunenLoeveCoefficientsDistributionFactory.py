import openturns as ot
import math as m

class MetaModelDistributionFactory:
    def __init__(self):
        pass
    def build(self, sample):
        return ot.MetaModelAlgorithm.BuildDistribution(sample)

class KarhunenLoeveCoefficientsDistributionFactory:
    def __init__(self):
        pass
    def build(self, sample):

        # try Gaussian with fallback to histogram
        dimension = sample.getDimension()
        marginals = [None] * dimension
        level = ot.ResourceMap.GetAsScalar('MetaModelAlgorithm-PValueThreshold')
        for i in range(dimension):
            sample_i = sample.getMarginal(i)
            candidate = ot.NormalFactory().build(sample_i)
            pValue = ot.FittingTest.Kolmogorov(sample_i, candidate, level).getPValue()
            if pValue < level:
                candidate = ot.HistogramFactory().build(sample_i)
            marginals[i] = candidate
        distribution = ot.ComposedDistribution(marginals)

        # test independence with fallback to beta copula
        isIndependent = True
        for j in range(dimension):
            marginalJ = sample.getMarginal(j)
            for i in range(j + 1, dimension):
                testResult = ot.HypothesisTest.Spearman(sample.getMarginal(i), marginalJ)
                isIndependent = isIndependent and testResult.getBinaryQualityMeasure()
        if not isIndependent:
            betaCopula = ot.EmpiricalBernsteinCopula(sample, sample.getSize())
            distribution.setCopula(betaCopula)

        return distribution
