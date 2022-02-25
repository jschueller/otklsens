import openturns as ot
from .KarhunenLoeveFCEAlgorithm import *
import math as m

class KarhunenLoeveSensitivityAnalysis:
    def __init__(self, inputProcessSample, outputSample, blockIndices=None)
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        self.blockIndices_ = blockIndices

    def run(self):
        algo = KarhunenLoeveFCEAlgorithm(self.inputProcessSample_, outputSample, blockIndices)
        algo.run()
