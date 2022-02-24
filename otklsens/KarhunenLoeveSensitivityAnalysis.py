import openturns as ot
from .EmpiricalKarhunenLoeveAlgorithm import *
from .EmpiricalKarhunenLoeveResult import *
import math as m

class KarhunenLoeveSensitivityAnalysis:
    def __init__(self, inputProcessSample, outputProcessSample, inputEpsilon=1e-3, outputEpsilon=1e-3, factories=list(), basisSize = 100, sparse = True, anisotropic = False):
        self.inputProcessSample_ = inputProcessSample
        self.outputProcessSample_ = outputProcessSample
        self.inputEpsilon_ = inputEpsilon
        self.outputEpsilon_ = outputEpsilon
        if len(factories) == 0:
            self.factories_ = [ot.HistogramFactory()]*inputProcessSample.getDimension()
        else:
            self.factories_ = factories
        self.basisSize_ = basisSize
        self.marginalInputSizes_ = ot.Indices()
        self.marginalOutputSizes_ = ot.Indices()
        self.sparse_ = sparse
        self.anisotropic_ = anisotropic
        self.inputEKLResult_ = EmpiricalKarhunenLoeveResult()
        self.outputEKLResult_ = EmpiricalKarhunenLoeveResult()
        self.marginalInputSizes_ = ot.Indices()
        self.marginalOutputSizes_ = ot.Indices()
        self.pceResult_ = ot.FunctionalChaosResult()
        self.allInputVariances_ = ot.Point()
        self.allOutputVariances_ = ot.Point()

    def findSignificantComponents(self, values, epsilon):
        dimension = values.getDimension()
        small = values[0] * epsilon
        # Can do better with bisection search...
        for i in range(dimension):
            if (values[i] <= small):
                indices = ot.Indices(i)
                indices.fill()
                return indices
        indices = ot.Indices(dimension)
        indices.fill()
        return indices

    def aggregateCoefficients(self, eklResult, epsilon):
        # Extract the significant components
        fullSample = ot.Sample(eklResult.getMarginalCoefficients(0).getSize(), 0)
        allSize = ot.Indices()
        allVariances = ot.Point()
        last = 0
        dimension = eklResult.getSize()
        for i in range(dimension):
            variances = eklResult.getMarginalVariances(i)
            indices = self.findSignificantComponents(variances, epsilon)
            fullSample.stack(eklResult.getMarginalCoefficients(i).getMarginal(indices))
            last += indices.getSize()
            allSize.add(last)
            allVariances.add(ot.Point([variances[i] for i in indices]))
        return fullSample, allSize, allVariances

    def computePCE(self, inSample, outSample):
        # Iterate over the input dimension and for each input dimension within the KL coefficients
        dimension = inSample.getDimension()
        j0 = 0
        # For the weights of the enumerate function
        weights = ot.Point(0)
        allInputVariances = self.allInputVariances_
        # For the input distribution
        coll = ot.DistributionCollection(0)
        # For the orthogonal basis
        polyColl = ot.PolynomialFamilyCollection(0)
        index = 0
        for i in range(self.marginalInputSizes_.getSize()):
            j1 = self.marginalInputSizes_[i]
            sigma0 = allInputVariances[j0]
            for j in range(j0, j1):
                print("i=", i, "j=", j, "index=", index)
                weights.add(m.sqrt(sigma0 / allInputVariances[j]))
                marginalDistribution = self.factories_[i].build(inSample.getMarginal(index))
                coll.add(marginalDistribution)
                polyColl.add(ot.StandardDistributionPolynomialFactory(marginalDistribution))
                index += 1
            j0 = j1
        # Build the distribution
        distribution = ot.ComposedDistribution(coll)
        # Build the enumerate function
        if self.anisotropic_:
            enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(weights, 1.0)
        else:
            enumerateFunction = ot.LinearEnumerateFunction(dimension)
        # Build the basis
        productBasis = ot.OrthogonalProductPolynomialFactory(
            polyColl, enumerateFunction)
        print("distribution dimension=", distribution.getDimension())
        print("enumerate dimension=", enumerateFunction.getDimension())
        print("basis size=", polyColl.getSize())
        # run algorithm
        adaptiveStrategy = ot.FixedStrategy(productBasis, self.basisSize_)
        if self.sparse_:
            projectionStrategy = ot.LeastSquaresStrategy(ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut()))
        else:
            projectionStrategy = ot.LeastSquaresStrategy()
        algo = ot.FunctionalChaosAlgorithm(
            inSample, outSample, distribution, adaptiveStrategy, projectionStrategy)
        algo.run()
        return algo.getResult()

    def run(self):
        # Build the empirical KL expansion of both input and output process samples
        inputAlgo = EmpiricalKarhunenLoeveAlgorithm(self.inputProcessSample_)
        inputAlgo.run()
        self.inputEKLResult_ = inputAlgo.getResult()
        fullInputSample, self.marginalInputSizes_, self.allInputVariances_ = self.aggregateCoefficients(self.inputEKLResult_, self.inputEpsilon_)
        outputAlgo = EmpiricalKarhunenLoeveAlgorithm(self.outputProcessSample_)
        outputAlgo.run()
        self.outputEKLResult_ = outputAlgo.getResult()
        fullOutputSample, self.marginalOutputSizes_, self.allOutputVariances_ = self.aggregateCoefficients(self.outputEKLResult_, self.outputEpsilon_)
        # Build the PCE decomposition of the link using Hermite polynomials
        self.pceResult_ = self.computePCE(fullInputSample, fullOutputSample)

    def computeSobolIndex(self, i, j):
        # Here we have to sum all the contributions of the coefficients of the PCE
        # that contributes to any of the coefficients of the jth marginal of the
        # output process.
        variance = 0.0
        conditionalVariance = 0.0
        # Get the range of input and output indices corresponding to the input marginal process i and the output marginal process j
        # Input
        startInput = 0
        if i > 0:
            startInput = self.marginalInputSizes_[i - 1]
        stopInput = self.marginalInputSizes_[i]
        # Output
        startOutput = 0
        if j > 0:
            startOutput = self.marginalOutputSizes_[j - 1]
        stopOutput = self.marginalOutputSizes_[j]
        # Now, select the relevant coefficients
        coefficients = self.pceResult_.getCoefficients()
        size = coefficients.getSize()
        enumerateFunction = self.pceResult_.getOrthogonalBasis().getEnumerateFunction()
        coefficientIndices = self.pceResult_.getIndices()
        for outputIndex in range(startOutput, stopOutput):
            for coeffIndex in range(size):
                coeff = coefficients[coeffIndex, outputIndex]
                # Only non-zero coefficients have to be taken into account
                if (coeff != 0.0):
                    k2 = coeff * coeff
                    variance += k2
                    # The only multi-indices we must take into account for
                    # the conditional variance are those associated to
                    # multi-indices that contain positive indices in the
                    # correct input range and null indices outside of this range
                    multiIndices = enumerateFunction(coefficientIndices[coeffIndex])
                    # Check if there is an index before the allowed range
                    isProperSubset = True
                    for k in range(startInput):
                        if (multiIndices[k] > 0):
                            isProperSubset = False
                            break
                    if not isProperSubset:
                        continue
                    # Check if there is an index after the allowed range
                    for k in range(stopInput, multiIndices.getSize()):
                        if (multiIndices[k] > 0):
                            isProperSubset = False
                            break
                    if not isProperSubset:
                        continue
                    for k in range(startInput, stopInput):
                        if (multiIndices[k] > 0):
                            conditionalVariance += k2
                        continue
        return conditionalVariance / variance

    def getInputEmpiricalKarhunenLoeveResult(self):
        return self.inputEKLResult_

    def getOutputEmpiricalKarhunenLoeveResult(self):
        return self.outputEKLResult_

    def getChaosResult(self):
        return self.pceResult_

    def drawVarianceDecomposition(self, j):
        # Here we have to sum all the contributions of the coefficients of the PCE
        # that contributes to any of the coefficients of the jth marginal of the
        # output process.
        # Get the range of input and output indices corresponding to the input marginal process i and the output marginal process j
        # Output
        startOutput = 0
        if j > 0:
            startOutput = self.marginalOutputSizes_[j - 1]
        stopOutput = self.marginalOutputSizes_[j]
        N = stopOutput - startOutput
        variances = ot.Sample(N, 1)
        coefficientIndices = self.pceResult_.getIndices()
        inputDimension = self.marginalInputSizes_.getSize()
        conditionalVariances = ot.Sample(N, inputDimension)
        # Now, select the relevant coefficients
        coefficients = self.pceResult_.getCoefficients()
        size = coefficients.getSize()
        enumerateFunction = self.pceResult_.getOrthogonalBasis().getEnumerateFunction()
        for outputIndex in range(N):
            for coeffIndex in range(size):
                coeff = coefficients[coeffIndex, startOutput + outputIndex]
                # Only non-zero coefficients have to be taken into account
                if (coeff != 0.0):
                    k2 = coeff * coeff
                    variances[outputIndex, 0] += k2
                    # The only multi-indices we must take into account for
                    # the conditional variance are those associated to
                    # multi-indices that contain positive indices in the
                    # correct input range and null indices outside of this range
                    multiIndices = enumerateFunction(coefficientIndices[coeffIndex])
                    for inputIndex in range(inputDimension):
                        startInput = 0
                        if inputIndex > 0:
                            startInput = self.marginalInputSizes_[inputIndex - 1]
                        stopInput = self.marginalInputSizes_[inputIndex]

                        # Check if there is an index before the allowed range
                        isProperSubset = True
                        for k in range(startInput):
                            if (multiIndices[k] > 0):
                                isProperSubset = False
                                break
                        if not isProperSubset:
                            continue
                        # Check if there is an index after the allowed range
                        for k in range(stopInput, multiIndices.getSize()):
                            if (multiIndices[k] > 0):
                                isProperSubset = False
                                break
                        if isProperSubset:
                            conditionalVariances[outputIndex, inputIndex] += k2
        conditionalVariances.exportToCSVFile("ConditionalVariances.csv")
        variances.exportToCSVFile("Variances.csv")
        graph = ot.Graph("Variance decomposition of output " + str(j), "KL mode", "variance", True, "topright")
        palette = ot.Drawable.BuildDefaultPalette(inputDimension + 1)
        lastSensitivities = ot.Point(inputDimension + 1)
        for outputIndex in range(N):
            y0 = 0.0
            for inputIndex in range(inputDimension):
                y1 = y0 + conditionalVariances[outputIndex, inputIndex]
                curve = ot.Curve([[outputIndex, y0], [outputIndex, y1]])
                curve.setColor(palette[inputIndex])
                curve.setLineWidth(3)
                if outputIndex == 0:
                    curve.setLegend("KL cond. var. " + str(inputIndex))
                else:
                    line = ot.Curve([[outputIndex - 1, lastSensitivities[inputIndex]], [outputIndex, y1]])
                    line.setLineStyle("dashed")
                    line.setColor(palette[inputIndex])
                    graph.add(line)
                graph.add(curve)
                y0 = y1
                lastSensitivities[inputIndex] = y1
            y1 = variances[outputIndex, 0]
            curve = ot.Curve([[outputIndex, y0], [outputIndex, y1]])
            curve.setColor(palette[inputDimension])
            curve.setLineWidth(3)
            if outputIndex == 0:
                curve.setLegend("KL tot. var.")
            else:
                line = ot.Curve([[outputIndex - 1, lastSensitivities[inputDimension]], [outputIndex, y1]])
                line.setLineStyle("dashed")
                line.setColor(palette[inputDimension])
                graph.add(line)
            graph.add(curve)
            lastSensitivities[inputDimension] = y1
        return graph
