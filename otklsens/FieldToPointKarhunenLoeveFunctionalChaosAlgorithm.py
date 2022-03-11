import openturns as ot
import math as m
from .KarhunenLoeveCoefficientsDistributionFactory import *

class FieldToPointKarhunenLoeveFunctionalChaosResult:
    def __init__(self, klResultCollection, marginalCoefficients, marginalVariances, fcaResult, inputProcessSample, outputSample, metamodel, residuals):
        self.klResultCollection_ = klResultCollection
        self.marginalCoefficients_ = marginalCoefficients
        self.marginalVariances_ = marginalVariances
        self.fcaResult_ = fcaResult
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        self.metamodel_ = metamodel
        self.residuals_ = residuals
    def getKarhunenLoeveResultCollection(self):
        return self.klResultCollection_
    def getMarginalCoefficients(self, i):
        return self.marginalCoefficients_[i]
    def getMarginalVariances(self, i):
        return self.marginalVariances_[i]
    def getFunctionalChaosResult(self):
        return self.fcaResult_
    def getInputProcessSample(self):
        return self.inputProcessSample_
    def getOutputSample(self):
        return self.outputSample_
    def getMetaModel(self):
        return self.metamodel_
    def getResiduals(self):
        return self.residuals_

class FieldToPointKarhunenLoeveFunctionalChaosAlgorithm:
    """
    KL/FCE-based field->vector metamodel.

    Parameters
    ----------
    inputProcessSample : :class:`openturns.ProcessSample`
        Input sample
    outputSample : 2-d sequence of float
        Output sample
    threshold : float, default=1e-3
        SVD decomposition eigenvalue threshold
    sparse : bool, default=True
        Whether to perform sparse or full FCE
    factory : :class:`openturns.DistributionFactory`
        Factory for the PCE on projected input process sample
    degree : int
        PCE degree
    """
    def __init__(self, inputProcessSample, outputSample, threshold=1e-3, sparse=True, blockIndices=None, factory=KarhunenLoeveCoefficientsDistributionFactory(), degree=2):
        if inputProcessSample.getSize() != outputSample.getSize():
            raise ValueError("input/output sample must have the same size")
        if blockIndices is None:
            blockIndices = list(range(inputProcessSample.getDimension()))
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        self.threshold_ = threshold
        self.blockIndices_ = blockIndices
        self.sparse_ = sparse
        self.result_ = None
        self.anisotropic_ = False
        self.factory_ = factory
        self.degree_ = degree

    def setSparse(sparse):
        self.sparse_ = sparse
    
    def getResult(self):
        return self.result_

    def computePCE(self, inSample, outSample):
        # Iterate over the input dimension and for each input dimension within the KL coefficients
        dimension = inSample.getDimension()
        #j0 = 0
        ## For the weights of the enumerate function
        #weights = ot.Point(0)
        #allInputVariances = self.allInputVariances_
        ## For the input distribution
        #coll = ot.DistributionCollection(0)
        ## For the orthogonal basis
        #polyColl = ot.PolynomialFamilyCollection(0)
        #index = 0
        #for i in range(self.marginalInputSizes_.getSize()):
            #j1 = self.marginalInputSizes_[i]
            #sigma0 = allInputVariances[j0]
            #for j in range(j0, j1):
                #print("i=", i, "j=", j, "index=", index)
                #weights.add(m.sqrt(sigma0 / allInputVariances[j]))
                #marginalDistribution = self.factories_[i].build(inSample.getMarginal(index))
                #coll.add(marginalDistribution)
                #polyColl.add(ot.StandardDistributionPolynomialFactory(marginalDistribution))
                #index += 1
            #j0 = j1
        # Build the distribution
        #distribution = ot.ComposedDistribution(coll)
        distribution = self.factory_.build(inSample)
        polyColl = [ot.StandardDistributionPolynomialFactory(distribution.getMarginal(i)) for i in range(dimension)]
        # Build the enumerate function
        #if self.anisotropic_:
            #enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(weights, 1.0)
        #else:
        enumerateFunction = ot.LinearEnumerateFunction(dimension)
        # Build the basis
        productBasis = ot.OrthogonalProductPolynomialFactory(polyColl, enumerateFunction)
        #print("distribution dimension=", distribution.getDimension())
        #print("enumerate dimension=", enumerateFunction.getDimension())
        #print("basis size=", polyColl.getSize())
        # run algorithm
        basisSize = m.comb(dimension + self.degree_, dimension)
        ot.Log.Info('dimension=' + str(dimension))
        ot.Log.Info('basisSize=' + str(basisSize))
        ot.Log.Info('outdim=' + str(outSample.getDimension()))
        adaptiveStrategy = ot.FixedStrategy(productBasis, basisSize)
        if self.sparse_:
            projectionStrategy = ot.LeastSquaresStrategy(ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut()))
        else:
            projectionStrategy = ot.LeastSquaresStrategy()
        algo = ot.FunctionalChaosAlgorithm(inSample, outSample, distribution, adaptiveStrategy, projectionStrategy)
        algo.run()
        return algo.getResult()

    def run(self):
        # build KL decomposition of input process sample
        inputDimension = self.inputProcessSample_.getDimension()
        centered = False
        algo = ot.KarhunenLoeveSVDAlgorithm(self.inputProcessSample_, self.threshold_, centered)
        algo.run()
        klResult = algo.getResult()

        # project input process sample
        projection = ot.KarhunenLoeveProjection(klResult)
        inputSample = projection(self.inputProcessSample_)

        # build PCE expansion of projected input sample vs output sample
        fcaResult = self.computePCE(inputSample, self.outputSample_)

        # compose input projection + PCE interpolation
        metamodel = ot.FieldToPointConnection(fcaResult.getMetaModel(), projection)

        # compute residual
        outputDimension = self.outputSample_.getDimension()
        residuals = [0.0] * outputDimension
        size = self.inputProcessSample_.getSize()
        for i in range(size):
            slack = metamodel(self.inputProcessSample_[i]) - self.outputSample_[i]
            for j in range(outputDimension):
                residuals[j] += slack[j] ** 2
        for j in range(outputDimension):
            residuals[j] = m.sqrt(residuals[j]) / size

        # marginal KL decomposition
        klResultCollection = []
        marginalCoefficients = []
        marginalVariances = []
        for d in range(inputDimension):
            inputProcessSample_i = self.inputProcessSample_.getMarginal(d)
            algo = ot.KarhunenLoeveSVDAlgorithm(inputProcessSample_i, self.threshold_, centered)
            algo.run()
            klResult_i = algo.getResult()
            projection = ot.KarhunenLoeveProjection(klResult_i)
            marginalCoefficients.append(projection(inputProcessSample_i))
            marginalVariances.append(ot.Point([sigma**2 for sigma in klResult_i.getEigenvalues()]))
            klResultCollection.append(klResult_i)

        self.result_ = FieldToPointKarhunenLoeveFunctionalChaosResult(klResultCollection, marginalCoefficients, marginalVariances, fcaResult, self.inputProcessSample_, self.outputSample_, metamodel, residuals)
