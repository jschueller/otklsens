import openturns as ot
from .KarhunenLoeveFCEAlgorithm import *
import math as m

class FieldToPointKarhunenLoeveSensitivityResult:
    def __init__(self):
        pass
    def getIndices(self):
        return [0]

class FieldToPointKarhunenLoeveSensitivityAnalysis:
    def __init__(self, inputProcessSample, outputSample, blockIndices=None):
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        if blockIndices is None:
            blockIndices = [list(range(inputProcessSample.getDimension()))]
        self.blockIndices_ = blockIndices
        self.result_ = None

    def getResult(self):
        return self.result_

    def run(self):
        algo = FieldToPointKarhunenLoeveFCEAlgorithm(self.inputProcessSample_, self.outputSample_)
        algo.run()
        result = algo.getResult()
        for bloc in self.blockIndices_:
            print('bloc', bloc)
    
