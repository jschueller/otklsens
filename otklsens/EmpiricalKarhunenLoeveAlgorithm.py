from openturns import *
from .EmpiricalKarhunenLoeveResult import *
from numpy import *
from time import *
from math import *

class EmpiricalKarhunenLoeveAlgorithm:

    def __init__(self, processSample):
        self.processSample_ = processSample
        self.result_ = EmpiricalKarhunenLoeveResult()
        
    def run(self):
        # We express the process X as:
        # X(omega, t) = mean(t) + \sum_{k=1}^d \sigma_k\xi_k(omega)\phi_k(t)
        # First, the mean function as a Field. We could recover a continuous
        # function using either interpolation, kriging or chaos
        marginalMean = list()
        marginalBasis = list()
        marginalCoefficients = list()
        marginalVariances = list()
        dimension = self.processSample_.getDimension()
        meanValues = self.processSample_.computeMean().getValues()
        ## print "meanValues=", meanValues
        mesh = self.processSample_.getMesh()
        nbFields = self.processSample_.getSize()
        nbVertices = mesh.getVerticesNumber()
        M = Matrix(nbVertices, nbFields)
        for d in range(dimension):
            marginalMean.append(Field(mesh, meanValues.getMarginal(d)))
            # Then we use SVD decomposition to get a sample of \xi and the
            # functions \phi_k
            # Build the matrix
            for j in range(nbFields):
                valuesJ = self.processSample_[j]
                for i in range(nbVertices):
                    M[i, j] = valuesJ[i, d] - meanValues[i, d]
            # Compute the SVD
            t0 = time()
            try:
                print("SVD")
                sigma, U, VT = M.computeSVD()
                nbColumns = U.getNbColumns()
                t1 = time()
                ## print "SVD=", t1 - t0, "s"
                ## print "sigma=", sigma, "U=\n", U
            except:
                print("SVD failed, use EV")
                lamb, ev = M.computeGram(False).computeEV()
                nbColumns = ev.getNbColumns()
                # We must sort the eigenvalues and the associated eigenvectors in descending order
                eigen_pairs = Sample(nbColumns, nbColumns + 1)
                for i in range(nbColumns):
                    for j in range(nbColumns):
                        eigen_pairs[i, j] = ev[i, j]
                    eigen_pairs[i, nbColumns] = -lamb[i]
                eigen_pairs = eigen_pairs.sortAccordingToAComponent(nbColumns)
                U = SquareMatrix(nbColumns)
                for i in range(nbColumns):
                    for j in range(nbColumns):
                        U[i, j] = eigen_pairs[i, j]
                    lamb[i] = -eigen_pairs[i, nbColumns]
                print("lamb=", lamb)
                sigma = Point([sqrt(max(0.0, lamb[i])) for i in range(nbColumns)])
                t1 = time()
            ## print "EV=", t1 - t0, "s"
            ## print "lamb=", lamb, "ev=\n", ev
            ## print "M=", M.getNbRows(), "x", M.getNbColumns()
            ## print "sigma=", sigma.getDimension()
            ## print "U=", U.getNbRows(), "x", U.getNbColumns()
            ## print U
            ## print "VT=", VT.getNbRows(), "x", VT.getNbColumns()
            ## print "nbVertices=", nbVertices
            ## print "sigma=", sigma, "alpha=", alpha
            marginalVariances.append(Point([sigma[i]**2 for i in range(sigma.getDimension())]))
            # Compute the coefficients
            coeffs = M.transpose() * U
            marginalCoefficients.append(Sample(array(coeffs)))
            basis = ProcessSample(mesh, 0, 1)
            # Extract the discretized basis functions
            for i in range(nbColumns):
                # Store the basis function
                col = U[:, i]
                values = Sample(array(col))
                basis.add(Field(mesh, values))
            marginalBasis.append(basis)
        self.result_ = EmpiricalKarhunenLoeveResult(marginalMean, marginalBasis, marginalCoefficients, marginalVariances)

    def getResult(self):
        return self.result_