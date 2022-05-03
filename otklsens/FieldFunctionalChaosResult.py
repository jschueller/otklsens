import openturns as ot

class FieldFunctionalChaosResult:
    def __init__(self, inputKLResultCollection, fceResult, outputKLResultCollection):
        self.inputKLResultCollection_ = inputKLResultCollection
        self.fceResult_ = fceResult
        self.outputKLResultCollection_ = outputKLResultCollection
        self.inputProcessSample_ = None
        self.outputSample_ = None
        self.blockIndices_ = None
        self.residuals_ = None
        self.fieldToPointMetamodel_ = None
        
    def setBlockIndices(self, blockIndices):
        self.blockIndices_ = blockIndices
    def getBlockIndices(self):
        return self.blockIndices_

    def getFieldToPointMetamodel(self):
        return self.fieldToPointMetamodel_
    def setMetamodel(self, metamodel):
        self.fieldToPointMetamodel_ = metamodel

    def getInputKLResultCollection(self):
        return self.inputKLResultCollection_
    def getFCEResult(self):
        return self.fceResult_
    def getOutputKLResultCollection(self):
        return self.outputKLResultCollection_

    def setInputProcessSample(self, inputProcessSample):
        self.inputProcessSample_ = inputProcessSample
    def getInputProcessSample(self):
        return self.inputProcessSample_

    def setOutputSample(self, outputSample):
        self.outputSample_ = outputSample
    def getOutputSample(self):
        return self.outputSample_

    def setResiduals(self, residuals):
        self.residuals = residuals
    def getResiduals(self):
        return self.residuals_
