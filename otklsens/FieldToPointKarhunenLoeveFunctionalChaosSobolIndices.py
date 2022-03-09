import openturns as ot
import math as m

class FieldToPointKarhunenLoeveFunctionalChaosSobolIndices:
    def __init__(self, result, blockIndices=None):
        self.result_ = result
        if blockIndices is None:
            blockIndices = [list(range(result.getInputProcessSample().getDimension()))]

    def getSobolIndex(self, i, j):
        
        return 0.0
    
