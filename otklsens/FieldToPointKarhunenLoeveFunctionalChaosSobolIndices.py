import openturns as ot
import math as m
import itertools

class FieldToPointKarhunenLoeveFunctionalChaosSobolIndices:
    def __init__(self, result):
        self.result_ = result
        kl_sizes = [len(result_i.getEigenvalues()) for result_i in result.getKarhunenLoeveResultCollection()]
        self.marginalInputSizes_ = list(itertools.accumulate(kl_sizes))

    def getSobolIndex(self, i, j):
        if i >= len(self.result_.getBlockIndices()):
            raise ValueError(f"Cannot ask for input index {i}")
        if j >= self.result_.getOutputSample().getDimension():
            raise ValueError(f"Cannot ask for output index {j}")
        blockIndices = self.result_.getBlockIndices()
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
    
