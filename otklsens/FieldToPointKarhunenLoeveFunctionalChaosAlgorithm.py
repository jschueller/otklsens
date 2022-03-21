import openturns as ot
import math as m
from .KarhunenLoeveCoefficientsDistributionFactory import *
from .StackedFieldToPointFunction import *

class FieldToPointKarhunenLoeveFunctionalChaosResult:
    def __init__(self, klResultCollection, fcaResult, inputProcessSample, outputSample, blockIndices, metamodel, residuals):
        self.klResultCollection_ = klResultCollection
        self.fcaResult_ = fcaResult
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        self.blockIndices_ = blockIndices
        self.metamodel_ = metamodel
        self.residuals_ = residuals
    def getKarhunenLoeveResultCollection(self):
        return self.klResultCollection_
    def getFunctionalChaosResult(self):
        return self.fcaResult_
    def getInputProcessSample(self):
        return self.inputProcessSample_
    def getOutputSample(self):
        return self.outputSample_
    def getBlockIndices(self):
        return self.blockIndices_
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
    basisSize : int
        PCE basis size
    """
    def __init__(self, inputProcessSample, outputSample, blockIndices=None, threshold=1e-3, sparse=True, factory=KarhunenLoeveCoefficientsDistributionFactory(), basisSize=100):
        if inputProcessSample.getSize() != outputSample.getSize():
            raise ValueError("input/output sample must have the same size")
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        if blockIndices is None:
            blockIndices = [[j] for j in range(inputProcessSample.getDimension())]
        self.blockIndices_ = blockIndices
        self.threshold_ = threshold
        self.sparse_ = sparse
        self.result_ = None
        self.anisotropic_ = False
        self.factory_ = factory
        self.basisSize_ = basisSize

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
        # run algorithm
        adaptiveStrategy = ot.FixedStrategy(productBasis, self.basisSize_)
        if self.sparse_:
            projectionStrategy = ot.LeastSquaresStrategy(ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut()))
        else:
            projectionStrategy = ot.LeastSquaresStrategy()
        algo = ot.FunctionalChaosAlgorithm(inSample, outSample, distribution, adaptiveStrategy, projectionStrategy)
        algo.run()
        return algo.getResult()

    def run(self):
        # build marginal KL decompositions of input process sample
        inputDimension = self.inputProcessSample_.getDimension()
        size = self.inputProcessSample_.getSize()
        klResultCollection = []
        projectionCollection = []
        inputSample = ot.Sample(size, 0)
        for i in range(len(self.blockIndices_)):
            blockIndices_i = self.blockIndices_[i]
            inputProcessSample_i = self.inputProcessSample_.getMarginal(blockIndices_i)
            centered = True
            algo = ot.KarhunenLoeveSVDAlgorithm(inputProcessSample_i, self.threshold_, centered)
            algo.run()
            klResult_i = algo.getResult()
            projection_i = ot.KarhunenLoeveProjection(klResult_i)
            projectionCollection.append(projection_i)
            inputSample.stack(projection_i(inputProcessSample_i))
            klResultCollection.append(klResult_i)

        # input process sample projection (+ reorder by blocks)
        py2f = StackedFieldToPointFunction(projectionCollection, self.blockIndices_)
        projection = ot.FieldToPointFunction(py2f)

        # build PCE expansion of projected input sample vs output sample
        fcaResult = self.computePCE(inputSample, self.outputSample_)
        print(f"PCE relative error={fcaResult.getRelativeErrors()}")

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
        
        self.result_ = FieldToPointKarhunenLoeveFunctionalChaosResult(klResultCollection, fcaResult, self.inputProcessSample_, self.outputSample_, self.blockIndices_, metamodel, residuals)
