import openturns as ot

class AggregatedFieldToPointFunction(ot.OpenTURNSPythonFieldToPointFunction):

    def __init__(self, coll):
        # first argument:
        outDim = sum([f.getOutputDimension() for f in coll])
        assert len(coll) > 0, "not a sequence"
        inDim = coll[0].getInputDimension()
        inDesc = coll[0].getInputDescription()
        outDesc = ot.Description()
        inputBlocs = []
        index = 0
        for f in coll:
            inputDim = f.getInputDimension()
            inputBlocs.append(list(range(index, inputDim)))
            #inDesc.add(f.getInputDescription())
            outDesc.add(f.getOutputDescription())
            index += inputDim
        self.inputBlocs_ = inputBlocs
        self.coll_ = coll
        mesh = coll[0].getInputMesh()
        super(AggregatedFieldToPointFunction, self).__init__(mesh, inDim, outDim)
        self.setInputDescription(inDesc)
        self.setOutputDescription(outDesc)

    def _exec(self, X):
        print(X)
        Xs = ot.Sample(X)
        Y = ot.Point()
        for i in range(len(self.coll_)):
            x = Xs # .getMarginal(self.inputBlocs_[i])
            f = self.coll_[i]
            y = f(x)
            Y.add(y)
        return Y
