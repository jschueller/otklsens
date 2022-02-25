import openturns as ot


class KarhunenLoeveFCEResult:
    def __init__(self, metamodel):
        self.metamodel_ = metamodel
    def getMetaModel(self):
        return self.metamodel_

class KarhunenLoeveFCEAlgorithm:
    """
    KarhunenLoeve and FCE based field metamodel.

    Parameters
    ----------
    inputProcessSample : :class:`openturns.ProcessSample`
        Input sample
    outputSample : 2-d sequence of float
        Output sample
    blockIndices : 2-d sequence of int
        List of independent blocks indices
    """
    def __init__(self, inputProcessSample, outputSample, blockIndices=None):
        if inputProcessSample.getSize() != outputSample.getSize():
            raise ValueError("input/output sample must have the same size")
        if blockIndices is None:
            blockIndices = list(range(inputProcessSample.getDimension()))
        self.inputProcessSample_ = inputProcessSample
        self.outputSample_ = outputSample
        self.blockIndices_ = blockIndices_
        self.sparse_ = True
        self.result_ = None
    
    def setSparse(sparse):
        self.sparse_ = sparse
    
    def getResult(self):
        return self.result_

    def run(self):
        for block in blockIndices:
            inputProcessSample = inputProcessSample.getMarginal(block)
            
            
