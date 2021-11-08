import numpy as np
import sympy as sym
import time


'''def eigenValue(matrix):
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
        detPoly = sym.Poly(symMat.det())
        detCoeff = detPoly.all_coeffs()
        lambda_ = np.round(np.roots(detCoeff),2)
        lambda_ = list(dict.fromkeys(lambda_))
        #print(detPoly)
        #print((detCoeff))
        #print(lambda_)
        #print(type(lambda_))
        return lambda_'''

def eigenValue(A, iterations=1000):
    Ak = np.copy(A)
    n = Ak.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        s = Ak.item(n-1, n-1)
        smult = s * np.eye(n)
        Q, R = np.linalg.qr(np.subtract(Ak, smult))
        Ak = np.add(np.matmul(R,Q), smult)
        QQ = np.matmul(QQ,Q)
        arrEigen = [0 for i in range (len(Ak))]
        x = 0
        for i in range(len(Ak)):
            for j in range(len(Ak[0])):
                if i == j :
                    arrEigen[x] = Ak[i][j]
                    x += 1
    arrEigen = np.round(arrEigen,4)
    arrEigen = list(dict.fromkeys(arrEigen))
    arrEigen.sort(reverse=True)
    print (arrEigen)

        
'''def eigenVector(matrix):
    arrEigen = eigenValue(matrix)
    print(arrEigen)
    row = len(matrix)
    col = len(matrix[0])
    # = sym.MatrixSymbol(["λ"])
    iMat = [[arrEigen[0] for i in range(col)] for j in range (row)]
    b = [[0] for i in range(row)]
    for i in range (row) :
        for j in range(col):
            if (i != j) :
                iMat[i][j] = 0

    iMat  = iMat - matrix
    #symMat = sym.Matrix(iMat)
    print(iMat)
    print(b)
    print(np.linalg.lstsq(iMat,b,rcond=None))
    #subbed = symMat.subs(x,5)
    #print(subbed)'''

def eigenVectorNorm(A):
    Q, R = np.linalg.qr(A) 
    Qbef = np.empty(shape=Q.shape)
    for i in range(100):
        Qbef[:] = Q
        X = np.matmul(A,Q)
        Q, R = np.linalg.qr(X)
        if np.allclose(Q, Qbef, atol=10e-4):
            break
    return Q
    
start_time = time.time()
A = np.array([[np.sqrt(12),0,0], [0,np.sqrt(10),0]])
B = np.array([[10,0,2], [0,10,4] ,[2,4,2]])
#A = np.random.random((100,100))
eigenValue(B)
print(eigenVectorNorm(B))
#print("--- %s seconds ---" % (time.time() - start_time))
#print(np.linalg.eigvals(A))

