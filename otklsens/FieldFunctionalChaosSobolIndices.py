import openturns as ot
import math as m
import itertools

class FieldFunctionalChaosSobolIndices:
    def __init__(self, result, verbose=False):
        self.result_ = result
        kl_sizes = [len(result_i.getEigenvalues()) for result_i in result.getInputKLResultCollection()]
        self.verbose_ = verbose
        if self.verbose_:
            ot.Log.Info(f"-- FieldFunctionalChaosSobolIndices kl_sizes={kl_sizes}")
        self.marginalInputSizes_ = list(itertools.accumulate(kl_sizes))

    def getSobolIndex(self, i, j):
        if i >= len(self.result_.getBlockIndices()):
            raise ValueError(f"Cannot ask for input index {i}")
        if j >= self.result_.getOutputSample().getDimension():
            raise ValueError(f"Cannot ask for output index {j}")
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
        if self.verbose_:
            ot.Log.Info(f"-- getSobolIndex i={i} j={j} startInput={startInput} stopInput={stopInput}")
        # Now, select the relevant coefficients
        coefficients = self.result_.getFCEResult().getCoefficients()
        size = coefficients.getSize()
        enumerateFunction = self.result_.getFCEResult().getOrthogonalBasis().getEnumerateFunction()
        coefficientIndices = self.result_.getFCEResult().getIndices()
        if self.verbose_:
            ot.Log.Info(f"-- coefficients={coefficients}")
        for coeffIndex in range(size):
            coeff = coefficients[coeffIndex, j]
            # Only non-zero coefficients have to be taken into account
            if (coeff != 0.0):
                k2 = coeff * coeff
                variance += k2
                # The only multi-indices we must take into account for
                # the conditional variance are those associated to
                # multi-indices that contain positive indices in the
                # correct input range and null indices outside of this range
                multiIndices = enumerateFunction(coefficientIndices[coeffIndex])
                if self.verbose_:
                    ot.Log.Info(f"-- coeffIndex={coeffIndex} multiIndices={multiIndices}")
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
    
