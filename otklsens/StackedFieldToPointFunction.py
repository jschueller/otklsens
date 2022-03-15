import openturns as ot

class StackedFieldToPointFunction(ot.OpenTURNSPythonFieldToPointFunction):

    def __init__(self, coll):
        # first argument:
        outDim = sum([f.getOutputDimension() for f in coll])
        inDim = sum([f.getInputDimension() for f in coll])
        inDesc = ot.Description()
        outDesc = ot.Description()
        inputBlocs = []
        index = 0
        for f in coll:
            inputDim = f.getInputDimension()
            inputBlocs.append(list(range(index, index+inputDim)))
            inDesc.add(f.getInputDescription())
            outDesc.add(f.getOutputDescription())
            index += inputDim
        self.inputBlocs_ = inputBlocs
        self.coll_ = coll
        mesh = coll[0].getInputMesh()
        super(StackedFieldToPointFunction, self).__init__(mesh, inDim, outDim)
        self.setInputDescription(inDesc)
        self.setOutputDescription(outDesc)

    def _exec(self, X):
        Xs = ot.Sample(X)
        #print('-- StackedFieldToPointFunction _exec X=', Xs.getDimension())
        Y = ot.Point()
        for i in range(len(self.coll_)):
            x = Xs.getMarginal(self.inputBlocs_[i])
            #print('-- StackedFieldToPointFunction _exec x=', x.getDimension())
            f = self.coll_[i]
            #print('-- StackedFieldToPointFunction _exec', f.getInputDimension(), f.getOutputDimension())
            y = f(x)
            Y.add(y)
        return Y
