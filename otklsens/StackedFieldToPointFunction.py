import openturns as ot

class StackedFieldToPointFunction(ot.OpenTURNSPythonFieldToPointFunction):

    def __init__(self, coll, blockIndices):
        outDim = sum([f.getOutputDimension() for f in coll])
        inDim = sum([f.getInputDimension() for f in coll])
        inDesc = ot.Description()
        outDesc = ot.Description()
        for f in coll:
            inDesc.add(f.getInputDescription())
            outDesc.add(f.getOutputDescription())
        self.blockIndices_ = blockIndices
        self.coll_ = coll
        mesh = coll[0].getInputMesh()
        super(StackedFieldToPointFunction, self).__init__(mesh, inDim, outDim)
        self.setInputDescription(inDesc)
        self.setOutputDescription(outDesc)

    def _exec(self, X):
        Xs = ot.Sample(X)
        Y = ot.Point()
        for i in range(len(self.coll_)):
            x = Xs.getMarginal(self.blockIndices_[i])
            f = self.coll_[i]
            y = f(x)
            Y.add(y)
        return Y
