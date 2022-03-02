import openturns as ot
import math as m

class MetamodelDistributionFactory:
    def __init__(self):
        pass
    def build(self, sample):
        return ot.MetaModelAlgorithm.BuildDistribution(sample)

class KLCoefficientsDistributionFactory:
    def __init__(self):
        pass
    def build(self, sample):
      
        # try standard PCE distributions
        factories = [ot.UniformFactory(), ot.NormalFactory(), ot.GammaFactory(), ot.BetaFactory()]
        dimension = sample.getDimension()
        marginals = [None] * dimension
        for i in range(dimension):
            sample_i = sample.getMarginal(i)
            candidates = [factory.build(sample_i) for factory in factories]
            marginals[i], testResult = ot.FittingTest.BestModelBIC(sample_i, candidates)
            cname = marginals[i].getImplementation().getClassName()
            if cname != 'Normal':
                raise ValueError('not Gaussian') 
        #ot.Log.Info(','.join([marginals[i].getImplementation().getClassName() for i in range(dimension)]))
        distribution = ot.ComposedDistribution(marginals)

        # test independence
        isIndependent = True
        for j in range(dimension):
            marginalJ = sample.getMarginal(j)
            for i in range(j + 1, dimension):
                testResult = ot.HypothesisTest.Spearman(sample.getMarginal(i), marginalJ)
                isIndependent = isIndependent and testResult.getBinaryQualityMeasure()
        if not isIndependent:
            #distribution.setCopula(ot.NormalCopulaFactory().build(sample));
            raise ValueError('not independent') 
    
        return distribution

class KarhunenLoeveFCEResult:
    def __init__(self, metamodel, residuals):
        self.metamodel_ = metamodel
        self.residuals_ = residuals
    def getMetaModel(self):
        return self.metamodel_
    def getResiduals(self):
        return self.residuals_



class FieldToPointKarhunenLoeveFCEAlgorithm:
    """
    KarhunenLoeve and FCE based field metamodel.

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
    def __init__(self, inputProcessSample, outputSample, threshold=1e-3, sparse=True, blockIndices=None, factory=KLCoefficientsDistributionFactory(), degree=2):
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
        # build KL expansion of input process sample
        algo = ot.KarhunenLoeveSVDAlgorithm(self.inputProcessSample_, self.threshold_)
        algo.run()
        klResult = algo.getResult()

        # project input process sample
        projection = ot.KarhunenLoeveProjection(klResult)
        inputSample = projection(self.inputProcessSample_)

        # build PCE expansion of projected input sample vs output sample
        pceResult = self.computePCE(inputSample, self.outputSample_)

        # compose input projection + PCE interpolation
        metamodel = ot.FieldToPointConnection(pceResult.getMetaModel(), projection)

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

        self.result_ = KarhunenLoeveFCEResult(metamodel, residuals)
