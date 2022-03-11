import openturns as ot
import math as m

class FieldToPointKarhunenLoeveFunctionalChaosSobolIndices:
    def __init__(self, result, blockIndices=None):
        self.result_ = result
        if blockIndices is None:
            blockIndices = [list(range(result.getInputProcessSample().getDimension()))]
        self.alreadyComputedAggregatedCoefficients_ = False

    def aggregateCoefficients(self, epsilon=1e-3):
        fullSample = ot.Sample(self.result_.getMarginalCoefficients(0).getSize(), 0)
        allSize = ot.Indices()
        allVariances = ot.Point()
        last = 0
        size = len(self.result_.getKarhunenLoeveResultCollection())
        for i in range(size):
            variances = self.result_.getMarginalVariances(i)
            small = variances[0] * epsilon
            Ki = len(variances)
            for j in range(Ki):
                if variances[j] < small:
                    Ki = j + 1
                    break
            indices = ot.Indices(Ki)
            indices.fill()
            fullSample.stack(self.result_.getMarginalCoefficients(i).getMarginal(indices))
            last += indices.getSize()
            allSize.add(last)
            allVariances.add(variances[indices])
        return fullSample, allSize, allVariances


    def getSobolIndex(self, i, j):
      
        if not self.alreadyComputedAggregatedCoefficients_:
            self.fullInputSample_, self.marginalInputSizes_, _ = self.aggregateCoefficients()

        #coefficients = self.result_.getFunctionalChaosResult().getCoefficients()
        #size = coefficients.getSize()
        
        #enumerateFunction = self.result_.getFunctionalChaosResult().getOrthogonalBasis().getEnumerateFunction()
        #coefficientIndices = self.result_.getFunctionalChaosResult().getIndices()
        #inputDimension = self.result_.getInputProcessSample().getDimension()
        #for i in range(inputDimension):
            #marginalCoefficients = self.result_.getMarginalCoefficients(i)
            #marginalVariances = self.result_.getMarginalVariances(i)
            #print('mc=', marginalCoefficients.getSize(), marginalCoefficients.getDimension())
            
        #fullInputSample, self.marginalInputSizes_, self.allInputVariances_
        
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
        outputIndex = j
        # Now, select the relevant coefficients
        coefficients = self.result_.getFunctionalChaosResult().getCoefficients()
        size = coefficients.getSize()
        enumerateFunction = self.result_.getFunctionalChaosResult().getOrthogonalBasis().getEnumerateFunction()
        coefficientIndices = self.result_.getFunctionalChaosResult().getIndices()
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
    
