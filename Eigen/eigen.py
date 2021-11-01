import numpy as np
import sympy as sym

def eigenValue(matrix):
        row = len(matrix)
        col = len(matrix[0])
        lamda = sym.symbols(["λ"])
        iMat = [lamda*col for i in range(row)]
        for i in range (row) :
            for j in range(col):
                if (i != j) :
                    iMat[i][j] = 0

        iMat  = iMat - matrix
        symMat = sym.Matrix(iMat)

        print(symMat.det())
        
        
        


A = np.array([[3,-2,0], [-2,3,0], [0,0,5]])
eigenValue(A)

                

