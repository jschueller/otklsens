import openturns as ot
import math as m
from .FieldFunctionalChaosResult import *

class StackedProjectionFunction(ot.OpenTURNSPythonFieldToPointFunction):

    def __init__(self, coll, blockIndices):
        outDim = sum([f.getOutputDimension() for f in coll])
        inDim = sum([f.getInputDimension() for f in coll])
        inDesc = ot.Description()
        outDesc = ot.Description()
        for f in coll:
            inDesc.add(f.getInputDescription())
            outDesc.add(f.getOutputDescription())
        self.blockIndices_ = blockIndices
        self.coll_ = coll
        mesh = coll[0].getInputMesh()
        super(StackedProjectionFunction, self).__init__(mesh, inDim, outDim)
        self.setInputDescription(inDesc)
        self.setOutputDescription(outDesc)

    def _exec(self, X):
        Xs = ot.Sample(X)
        Y = ot.Point()
        for i in range(len(self.coll_)):
            x = Xs.getMarginal(self.blockIndices_[i])
            f = self.coll_[i]
            y = f(x)
            Y.add(y)
        return Y

class FieldToPointFunctionalChaosAlgorithm:
    """
    KL/FCE-based field->vector metamodel.

    Parameters
    ----------
    inputProcessSample : :class:`openturns.ProcessSample`
        Input sample
    outputSample : 2-d sequence of float
        Output sample
    """
    def __init__(self, inputProcessSample, outputSample):
        if inputProcessSample.getSize() != outputSample.getSize():
            raise ValueError("input/output sample must have the same size")
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        inputDimension = inputProcessSample.getDimension()
        self.blockIndices_ = [[j] for j in range(inputDimension)]
        flat = ot.Indices()
        for block in self.blockIndices_:
            flat.add(ot.Indices(block))
        if flat.getSize() != inputDimension or not flat.check(inputDimension):
            raise ValueError("invalid block indices")
        self.threshold_ = 0.0
        self.nbModes_ = 2**30
        self.sparse_ = True
        self.result_ = None
        self.anisotropic_ = False
        self.basisSize_ = 100
        self.recompress_ = False

    def setThreshold(self, threshold):
        """
        KL spectrum cut-off threshold.

        Parameters
        ----------
        threshold : float
            KL decomposition spectrum cut-off threshold
            Both for input blocs decomposition and recompression if enabled
        """
        self.threshold_ = threshold

    def setNbModes(self, nbModes):
        """
        KL max modes number.

        Parameters
        ----------
        nbModes : int
            Maximum number of KL modes
            Both for input blocs decomposition and recompression if enabled
        """
        self.nbModes_ = nbModes

    def setBasisSize(self, basisSize):
        """
        PCE basis size accessor.

        Parameters
        ----------
        basisSize : int
            PCE basis size
        """
        self.basisSize_ = basisSize

    def setSparse(self, sparse):
        """
        Sparse chaos flag accessor.
        
        Parameters
        ----------
        sparse : bool
            Whether to perform sparse or full FCE
        """
        self.sparse_ = sparse

    def setRecompress(self, recompress):
        """
        Recompression flag accessor.
        
        Parameters
        ----------
        recompress : bool, default=False
            Whether to eliminate more modes in the global list
        """
        self.recompress_ = recompress

    def setBlockIndices(self, blockIndices):
        """
        Block indices accessor.
        """
        self.blockIndices_ = blockIndices

    def getResult(self):
        return self.result_

    @staticmethod
    def BuildDistribution(sample):
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
        distribution = FieldToPointFunctionalChaosAlgorithm.BuildDistribution(inSample)
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
            sumPart = listEv[0]
            K = 1
            nbModesMax = min(self.nbModes_, len(listEv))
            while (K < nbModesMax) and (sumPart < (1.0 - self.threshold_) * sumEv):
                sumPart += listEv[K]
                K += 1
            lambdaT = listEv[K]
            ot.Log.Info(f"keep K={K}/{len(listEv)} modes, lambda threshold={lambdaT}")

            for i in range(len(self.blockIndices_)):
                ev_i = klResultCollection[i].getEigenvalues()
                # count number of modes to keep
                Ki = 1
                while (Ki < len(ev_i)) and (ev_i[Ki] >= lambdaT):
                    Ki += 1
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
        py2f = StackedProjectionFunction(projectionCollection, self.blockIndices_)
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
        
        result = FieldFunctionalChaosResult(klResultCollection, fceResult, [])
        result.setBlockIndices(self.blockIndices_)
        result.setInputProcessSample(self.inputProcessSample_)
        result.setOutputSample(self.outputSample_)
        result.setMetamodel(metamodel)
        result.setResiduals(residuals)
        self.result_ = result
