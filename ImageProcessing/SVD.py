import numpy as np
from numpy.linalg.linalg import eigvals
import sympy as sym

'''
Matriks A (m x n) dijadiin matriks U, Sigma, Vtrans

U = Matriks orto m x m
Sigma = Matriks m x n elemen diagonal utama adalah nilai singular (sqrt of eigen vals) 
V = Matriks orto n x n

FOR U
1. A * Atrans
2. find nilai eigen --> make matrix from tot of non-zero eigen vals
3. find vektor eigen
4. normalize vectors of eigenspace
5. fill columns with vectors of eigenspace

FOR V
1. Atrans * A
2. find nilai eigen --> make matrix from tot of ALL eigen vals
3. find vektor eigen
4. normalize vectors of eigenspace
5. transpose

FOR SIGMA
1. check Atrans * A's nilai singular (sqrt of non-zero eigen vals)
2. m x n matrix filled with 0
3. fill main diagonal with singular values from Atrans * A (big to little)
'''
def getLeftSingular(matrix):
    # initialization of u matrix (m x m) --> skarang gapake krn eigenVec tinggal transpose
    # nanti sesuaiin aja sama output dari eigenVec dhika

    # get eigen values and eigen vectors of a matrix
    a = np.dot(matrix, np.transpose(matrix))
    eigenVal, eigenVec = np.linalg.eig(a) # ini hasilnya uda dinormalize
    
    # normalize vectors
    # tambahin nanti

    # fill matrix
    u = np.transpose(eigenVec)

    # check for values, nanti apus!!
    print("Eigen vector:\n", eigenVec)
    print("Eigen values: ", eigenVal)
    print("Left Singular: \n", u)


def getRightSingular(matrix):
    # initialization of u matrix (m x m) --> skarang gapake krn eigenVec tinggal transpose
    # nanti sesuaiin aja sama output dari eigenVec dhika

    # get eigen values and eigen vectors of a matrix
    a = np.dot(np.transpose(matrix), matrix)
    eigenVal, eigenVec = np.linalg.eig(a) # ini hasilnya uda dinormalize
    
    # normalize vectors
    # tambahin nanti

    # fill matrix
    vtrans = eigenVec

    # check for values, nanti apus!!
    print("Eigen vector:\n", eigenVec)
    print("Eigen values: ", eigenVal)
    print("Right Singular: \n", vtrans)

def getSingularValues(matrix):
    #initialization of sigma matrix (m x n)
    m = len(matrix)
    n = len(matrix[0])
    sigma = [[0 for i in range(n)] for j in range(m)]

    # get eigen values and eigen vectors of matrix a
    a = np.dot(np.transpose(matrix), matrix)
    eigenVal = np.linalg.eigvals(a)

    # get singular values from eigen values
    singVal = []
    for i in eigenVal:
        if (i != 0):
            singVal.append(np.math.sqrt(i))
    singVal.sort(reverse=True)

    # fill main diagonal
    if (m < n):
        end = m
    else:
        end = n
    for i in range(end):
        sigma[i][i] = singVal[i]

    # check for values, nanti apus!!!
    print("Eigen values: ", eigenVal)
    print("Singular values: ", singVal)
    print("Matriks sigma:\n", sigma)

# testing implementation
matrix = [[3, 1, 1], [-1, 3, 1]]
getSingularValues(matrix)
getLeftSingular(matrix)
getRightSingular(matrix)