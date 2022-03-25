import openturns as ot
import math as m
from .KLCoefficientsDistributionFactory import *
from .StackedFieldToPointFunction import *

class FieldToPointKLFCEResult:
    def __init__(self, klResultCollection, fceResult, inputProcessSample, outputSample, blockIndices, metamodel, residuals):
        self.klResultCollection_ = klResultCollection
        self.fceResult_ = fceResult
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        self.blockIndices_ = blockIndices
        self.metamodel_ = metamodel
        self.residuals_ = residuals
    def getKLResultCollection(self):
        return self.klResultCollection_
    def getFCEResult(self):
        return self.fceResult_
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

class FieldToPointKLFCEAlgorithm:
    """
    KL/FCE-based field->vector metamodel.

    Parameters
    ----------
    inputProcessSample : :class:`openturns.ProcessSample`
        Input sample
    outputSample : 2-d sequence of float
        Output sample
    threshold : float, default=0.0
        KL decomposition spectrum cut-off threshold
        Both for input blocs decomposition and recompression if enabled
    nbModes : int, default=+inf
        Maximum number of KL modes
        Both for input blocs decomposition and recompression if enabled
    sparse : bool, default=True
        Whether to perform sparse or full FCE
    factory : :class:`openturns.DistributionFactory`
        Multivariate factory for the PCE on projected input process sample
    basisSize : int
        PCE basis size
    recompress : bool, default=False
        Whether to eliminate more modes in the global list
    """
    def __init__(self, inputProcessSample, outputSample, blockIndices=None, threshold=0.0, nbModes=2**30, sparse=True, factory=KLCoefficientsDistributionFactory(), basisSize=100, recompress=False):
        if inputProcessSample.getSize() != outputSample.getSize():
            raise ValueError("input/output sample must have the same size")
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        inputDimension = inputProcessSample.getDimension()
        if blockIndices is None:
            blockIndices = [[j] for j in range(inputDimension)]
        self.blockIndices_ = blockIndices
        flat = ot.Indices()
        for block in blockIndices:
            flat.add(ot.Indices(block))
        if flat.getSize() != inputDimension or not flat.check(inputDimension):
            raise ValueError("invalid block indices")
        self.threshold_ = threshold
        self.nbModes_ = nbModes
        self.sparse_ = sparse
        self.result_ = None
        self.anisotropic_ = False
        self.factory_ = factory
        self.basisSize_ = basisSize
        self.recompress_ = recompress

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
        allEv = ot.Point()
        for i in range(len(self.blockIndices_)):
            inputProcessSample_i = self.inputProcessSample_.getMarginal(self.blockIndices_[i])
            centered = True
            algo = ot.KarhunenLoeveSVDAlgorithm(inputProcessSample_i, self.threshold_, centered)
            algo.setNbModes(self.nbModes_)
            algo.run()
            klResult_i = algo.getResult()
            allEv.add(klResult_i.getEigenvalues())
            klResultCollection.append(klResult_i)
            ot.Log.Info(f"block#{i}={self.blockIndices_[i]} ev={klResult_i.getEigenvalues()}")

        if self.recompress_:
            sumEv = allEv.norm1()
            listEv = [ev for ev in allEv]
            listEv.sort(reverse=True)
            sumPart = 0.0
            K = 0
            nbModesMax = min(self.nbModes_, len(listEv))
            while (K < nbModesMax) and (listEv[K] >= self.threshold_ * sumEv):
                K += 1
            lambdaT = listEv[K]
            ot.Log.Info(f"keep K={K}/{len(listEv)} modes, lambda threshold={lambdaT}")
            for i in range(len(self.blockIndices_)):
                ev_i = klResultCollection[i].getEigenvalues()
                # count number of modes to keep
                Ki = 0
                while (Ki < len(ev_i)) and (ev_i[Ki] >= lambdaT):
                    Ki += 1
                # keep at least one mode
                if Ki == 0:
                    Ki = 1
                ot.Log.Info(f"i={i} keep Ki={Ki}/{len(ev_i)} modes")

                # keep only Ki first modes and rebuild result
                klResult_i = klResultCollection[i]
                selectedEV2 = klResult_i.getEigenvalues()[:Ki]
                modes2 = ot.FunctionCollection(Ki)
                for k in range(Ki):
                    modes2[k] = klResult_i.getModes()[k]
                modes2 = ot.Basis(modes2)
                covariance2 = ot.RankMCovarianceModel(selectedEV2, modes2)
                modesAsProcessSample = klResult_i.getModesAsProcessSample()
                # TODO use new ProcessSample.erase method
                modesAsProcessSample2 = ot.ProcessSample(modesAsProcessSample.getMesh(), Ki, modesAsProcessSample.getDimension())
                for k in range(Ki):
                    modesAsProcessSample2[k] = modesAsProcessSample[k]
                projectionMatrix2 = klResult_i.getProjectionMatrix()[:Ki,:]
                klResultCollection[i] = ot.KarhunenLoeveResult(covariance2, klResult_i.getThreshold(), selectedEV2, modes2, modesAsProcessSample2, projectionMatrix2)

        # the global projection stacks projections of each block of variables
        projectionCollection = []
        inputSample = ot.Sample(size, 0)
        for i in range(len(self.blockIndices_)):
            projection_i = ot.KarhunenLoeveProjection(klResultCollection[i])
            projectionCollection.append(projection_i)
            inputProcessSample_i = self.inputProcessSample_.getMarginal(self.blockIndices_[i])
            inputSample.stack(projection_i(inputProcessSample_i))
        py2f = StackedFieldToPointFunction(projectionCollection, self.blockIndices_)
        projection = ot.FieldToPointFunction(py2f)
        ot.Log.Info(f"total K={inputSample.getDimension()} modes")

        # build PCE expansion of projected input sample vs output sample
        fceResult = self.computePCE(inputSample, self.outputSample_)
        ot.Log.Info(f"PCE relative error={fceResult.getRelativeErrors()}")

        # compose input projection + PCE interpolation
        metamodel = ot.FieldToPointConnection(fceResult.getMetaModel(), projection)

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
        
        self.result_ = FieldToPointKLFCEResult(klResultCollection, fceResult, self.inputProcessSample_, self.outputSample_, self.blockIndices_, metamodel, residuals)
