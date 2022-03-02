
import openturns as ot

class LeastSquaresFCE:

    def __init__(self, X = None, wX = None, Y = None, distribution = None, basis = None, maximumBasisSize = None, leastSquaresMethod = "SVD"):
        self.X = X
        self.wX = wX
        self.Y = Y
        self.distribution = distribution
        self.basis = basis
        self.maximumBasisSize = maximumBasisSize
        self.leastSquaresMethod = leastSquaresMethod
        self.result = None

    def run(self):
        transformation = ot.DistributionTransformation(distribution, basis.getMeasure())
        XTransformed = transformation(self.X)
        indices = ot.Indices(self.maximumBasisSize)
        indices.fill()
        functions = [basis.build(i) for i in indices]
        designProxy = ot.DesignProxy(XTransformed, functions)
        if self.wX is None:
            leastSquaresMethod = ot.LeastSquaresMethod.Build(self.leastSquaresMethod, designProxy, indices)
        else:
            leastSquaresMethod = ot.LeastSquaresMethod.Build(self.leastSquaresMethod, designProxy, self.wX, indices)
        outputDimension = self.Y.getDimension()
        coefficients = ot.Sample(self.maximumBasisSize, outputDimension)
        if 0:
            for j in range(outputDimension):
                coeffsJ = leastSquaresMethod.solve(Y.getMarginal(j).asPoint())
                for i in range(self.maximumBasisSize):
                    coefficients[i, j] = coeffsJ[i]
        else:
            B = ot.Matrix(Y.getSize(), Y.getDimension())
            for i in range(Y.getSize()):
                for j in range(Y.getDimension()):
                    B[i, j] = Y[i, j]
            coeffs = leastSquaresMethod.solveMatrix(B)
            for j in range(outputDimension):
                for i in range(self.maximumBasisSize):
                    coefficients[i, j] = coeffs[i, j]
        self.result = ot.FunctionalChaosResult(ot.Function(), self.distribution, transformation, transformation.inverse(), ot.Function(), self.basis, indices, coefficients, functions, [-1.0], [-1.0])

    def getResult(self):
        if self.result is None:
            self.run()
        return self.result

if __name__ == "__main__":
    from math import pi
    a = 7.0
    b = 0.1
    inputVariables = ["xi1", "xi2", "xi3"]
    formula = ["sin(xi1) + (" + str(a) + \
        ") * (sin(xi2)) ^ 2 + (" + str(
            b) + ") * xi3^4 * sin(xi1)"]*10
    model = ot.SymbolicFunction(inputVariables, formula)

    # Create the input distribution
    dimension = len(inputVariables)
    distribution = ot.ComposedDistribution([ot.Uniform(-pi, pi)] * dimension)

    # Create the orthogonal basis
    enumerateFunction = ot.LinearEnumerateFunction(dimension)
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.LegendreFactory()] * dimension, enumerateFunction)

    # Create the input/output database
    size = 100000
    X = distribution.getSample(size)
    Y = model(X)
    basisSize = 500
    algo = LeastSquaresFCE(X = X, Y = Y, distribution = distribution, basis = basis, maximumBasisSize = basisSize)
    from time import time
    t0 = time()
    result = algo.getResult()
    t1 = time()
    #print("result=", result)
    print("t=", t1 - t0, "s")
    algo = ot.FunctionalChaosAlgorithm(X, Y, distribution, ot.FixedStrategy(basis, basisSize), ot.LeastSquaresStrategy())
    t0 = time()
    algo.run()
    result = algo.getResult()
    t1 = time()
    #print("result=", result)
    print("t=", t1 - t0, "s")

#before
#t= 24.17275619506836 s
#t= 181.62715983390808 s
