from openturns import *

class EmpiricalKarhunenLoeveResult:
    def __init__(self, marginalEmpiricalMean = list(), marginalEmpiricalBasis = list(), marginalCoefficients = list(), marginalVariances = list()):
        self.marginalEmpiricalMean_ = marginalEmpiricalMean
        self.marginalEmpiricalBasis_ = marginalEmpiricalBasis
        self.marginalCoefficients_ = marginalCoefficients
        self.marginalVariances_ = marginalVariances

    def getMarginalEmpiricalMean(self, i):
        return self.marginalEmpiricalMean_[i]

    def getMarginalEmpiricalBasis(self, i):
        return self.marginalEmpiricalBasis_[i]

    def getMarginalCoefficients(self, i):
        return self.marginalCoefficients_[i]

    def getMarginalVariances(self, i):
        return self.marginalVariances_[i]

    def getSize(self):
        return len(self.marginalEmpiricalMean_)

    def getMarginalMean(self, i):
        field = self.getMarginalEmpiricalMean(i)
        if field.getInputDimension() == 1:
            nbVertices = field.getMesh().getVerticesNumber()
            locations = Point([field.getMesh().getVertices()[i, 0] for i in range(nbVertices)])
            return Function(PiecewiseLinearEvaluation(locations, field.getValues()))
        return Function(field.getMesh().getVertices(), field.getValues())

    def getMarginalBasis(self, i):
        processSample = self.getMarginalEmpiricalBasis(i)
        nbVertices = processSample.getMesh().getVerticesNumber()
        locations = Point([processSample.getMesh().getVertices()[i, 0] for i in range(nbVertices)])
        size = processSample.getSize()
        coll = FunctionCollection(size)
        for i in range(size):
            values = processSample[i]
            if processSample.getMesh().getDimension() == 1:
                coll[i] = Function(PiecewiseLinearEvaluation(locations, values))
            else:
                coll[i] = Function(processSample.getMesh().getVertices(), values)
        return coll
        
