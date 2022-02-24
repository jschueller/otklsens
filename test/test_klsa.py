import openturns as ot
from otklsens import KarhunenLoeveSensitivityAnalysis
from math import *
from time import time
from openturns.viewer import View

def header(msg):
    t0 = time()
    print("="*len(msg))
    print(msg)
    print("="*len(msg))
    return(t0)

def footer(t0):
    print("t=", time() - t0, "s")

def test_klsa_big():
    t00 = time()
    # Verbosity
    ot.Log.Show(ot.Log.INFO)
    #Log.Show(Log.NONE)

    #############################################
    # Definition of the time grid and the model #
    #############################################
    t0 = header("Definition of the time grid and the model")

    # Time grid parameters
    T = 3.0
    NT = 32
    tg = ot.RegularGrid(0.0, T / NT, NT)

    # Toy function to link input processes to the output process
    in_dim = 4
    out_dim = 1
    spatial_dim = 1

    def myPyFunc(X):
        values = ot.Sample(X)
        f = ot.SymbolicFunction(["x1", "x2", "x3", "x4"], ["x1 + x2 + x3 - x4 + x1 * x2 - x3 * x4 - 0.1 * x1 * x2 * x3"])
        Y = f(values)
        return Y

    myFunc = ot.PythonFieldFunction(tg, in_dim, tg, out_dim, myPyFunc)

    footer(t0)
    ##################################################################
    # Creation of the input process as an aggregation of 3 processes #
    ##################################################################
    t0 = header("Creation of the input process as an aggregation of 3 processes")

    # First process: white noise around a bumpy trend
    # The bumpy trend
    refData = ot.CompositeProcess(ot.TrendTransform(ot.SymbolicFunction("t", "sin(20*t)"), tg), ot.RandomWalk([0.0], ot.Uniform(-0.2, 0.1), tg)).getRealization()
    vertices = refData.getMesh().getVertices()
    values = refData.getValues()

    refCurve = ot.DatabaseFunction(vertices, values)
    refCurve = ot.PiecewiseLinearEvaluation([vertices[i, 0] for i in range(vertices.getSize())], values)
    # The first process
    firstProcess = ot.CompositeProcess(ot.TrendTransform(refCurve, tg), ot.WhiteNoise(ot.Normal(0.0, 0.05), tg))
    firstProcess.setTimeGrid(tg)

    # Second process: smooth Gaussian process
    secondProcess = ot.GaussianProcess(ot.SquaredExponential([1.0], [T / 4.0]), tg)

    # Third process: elementary process based on a bivariate random vector
    randomParameters = ot.ComposedDistribution([ot.Uniform(), ot.Normal()])
    thirdProcess = ot.FunctionalBasisProcess(randomParameters, ot.Basis([ot.SymbolicFunction(["t"], ["1", "0"]), ot.SymbolicFunction(["t"], ["0", "1"])]))

    # Here we aggregate all the input processes
    processCollection = ot.ProcessCollection(0)
    processCollection.add(firstProcess)
    processCollection.add(secondProcess)
    processCollection.add(thirdProcess)
    X = ot.AggregatedProcess(processCollection)
    X.setMesh(tg)

    footer(t0)
    ###########################################################################
    # Creation of the database on which the sensitivity analysis will be done #
    ###########################################################################
    t0 = header("Creation of the database on which the sensitivity analysis will be done")

    # Sampling size
    size = 1024

    inSample = X.getSample(size)
    outSample = myFunc(inSample)
    print("inSample=", inSample.getSize(), inSample.getMesh().getVerticesNumber())
    print("outSample=", outSample.getSize(), outSample.getMesh().getVerticesNumber())

    footer(t0)
    ##################################################
    # Creation of the sensitivity analysis algorithm #
    ##################################################
    t0 = header("Creation of the sensitivity analysis algorithm")

    # Thresholds for the truncation of the Karhunen-Loeve (KL) expansions of the input and output samples
    input_KL_threshold  = 1.0e-3
    output_KL_threshold = 1.0e-3
    # Marginal distribution factories to use for the KL coefficients of the input process. Give one factory for each marginal input process, it will be used for all the coefficients of the KL expansion of the marginal input process
    input_coefficients_factories = [ot.NormalFactory(), ot.NormalFactory(), ot.ExponentialFactory(), ot.NormalFactory()]
    # Numerical parameters for the polynomial chaos expansion (PCE)
    # Numerical method to use for the least-squares approximation. Cholesky is the fastest, SVD the most robust
    ot.ResourceMap.SetAsString("LeastSquaresMetaModelSelection-DecompositionMethod", "Cholesky")
    # Early exit based on cross-validation error increase
    ot.ResourceMap.SetAsScalar("LeastSquaresMetaModelSelection-MaximumErrorFactor", 1.2)
    # Functional basis size.
    basisSize = 1000
    # Use sparse expansion?
    use_sparse = True
    # Use anisotropic exploration?
    use_anisotropic = True
    algo = KarhunenLoeveSensitivityAnalysis(inSample, outSample, input_KL_threshold, output_KL_threshold, input_coefficients_factories, basisSize, use_sparse, use_anisotropic)

    footer(t0)
    ####################################################
    # Computation of all the decompositions (KL + PCE) #
    ####################################################
    t0 = header("Computation of all the decompositions (KL + PCE)")
    algo.run()

    footer(t0)
    ######################################################
    # Post-processing: Sobol indices, graphical analysis #
    ######################################################
    t0 = header("Post-processing: Sobol indices, graphical analysis")
    print() 
    print("Sobol indices")
    for i in range(in_dim):
        print("Sobol(" + str(i) + ", 0)=", algo.computeSobolIndex(i, 0))

    print()
    print("Effective dimension of the input processes")
    i0 = 0
    for i in range(in_dim):
        i1 = algo.marginalInputSizes_[i]
        print("dim input(" + str(i) + ")=", i1 - i0)
        i0 = i1

    print()
    print("Effective dimension of the output processes")
    print("dim output(0)=", algo.marginalOutputSizes_[0])

    print()
    print("Draw the variance analysis")

    graph = algo.drawVarianceDecomposition(0)
    View(graph).save("VarianceDecomposition.pdf")

    print()
    print("Draw the input KL basis")
    inputEKLResult = algo.getInputEmpiricalKarhunenLoeveResult()
    size = inputEKLResult.getSize()
    for i in range(size):
        graph = inputEKLResult.getMarginalMean(i).draw(tg.getStart(), tg.getEnd())
        graph.setTitle("Mean of input " + str(i))
        #delta = sqrt(inputEKLResult.getMarginalVariances(i)[0])
        #bb[2] -= delta
        #bb[3] += delta
        View(graph).save("Input_" + str(i).zfill(2) + "_Mean.pdf")

    for i in range(size):
        basis = inputEKLResult.getMarginalBasis(i)
        N = basis.getSize()
        palette = ot.Drawable.BuildDefaultPalette(N)
        graph = ot.Graph("KL modes input " + str(i), "t", "phi", True, "topright")
        variances = inputEKLResult.getMarginalVariances(i)
        for j in range(N):
            curveJ = (basis[j] * ot.SymbolicFunction("x", str(sqrt(variances[j])))).draw(tg.getStart(), tg.getEnd()).getDrawable(0)
            curveJ.setLegend("KL " + str(j))
            curveJ.setColor(palette[j])
            graph.add(curveJ)
        View(graph).save("Input_" + str(i).zfill(2) + "_KLFunctions.pdf")

    print()
    print("Draw the output KL basis")
    outputEKLResult = algo.getOutputEmpiricalKarhunenLoeveResult()
    size = outputEKLResult.getSize()
    for i in range(size):
        graph = outputEKLResult.getMarginalMean(i).draw(tg.getStart(), tg.getEnd())
        graph.setTitle("Mean of output " + str(i))
        View(graph).save("Output_" + str(i).zfill(2) + "_Mean.pdf")

    for i in range(size):
        basis = outputEKLResult.getMarginalBasis(i)
        N = basis.getSize()
        palette = ot.Drawable.BuildDefaultPalette(N)
        graph = ot.Graph("KL modes output " + str(i), "t", "phi", True, "topright")
        variances = inputEKLResult.getMarginalVariances(i)
        for j in range(N):
            curveJ = (basis[j] * ot.SymbolicFunction("x", str(sqrt(variances[j])))).draw(tg.getStart(), tg.getEnd()).getDrawable(0)
            curveJ.setLegend("KL " + str(j))
            curveJ.setColor(palette[j])
            graph.add(curveJ)
        View(graph).save("Output_" + str(i).zfill(2) + "_KLFunctions.pdf")
    print()

    print("All significant input variances=", algo.allInputVariances_)
    print("All significant output variances=", algo.allOutputVariances_)
    footer(t0)

    print("Total")
    footer(t00)
