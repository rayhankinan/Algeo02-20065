import os # ini nanti ilangin cm buat testing
import cv2
import numpy as np
from Eigen import eigenValue, eigenVectorNorm

def getU(matrix): # left singular value
    # get eigen values and eigen vectors of a matrix
    a = np.dot(matrix, np.transpose(matrix))
    u = np.array(eigenVectorNorm(a)) # ini hasilnya uda dinormalize
    return u

def getVTranpose(matrix): # right singular value
    matrix_t = matrix.T
    # get eigen values and eigen vectors of a matrix
    a = np.dot(np.transpose(matrix), matrix)
    vtrans = np.array(eigenVectorNorm(a)) # ini hasilnya uda dinormalize
    return vtrans

def getSigma(matrix): # singular values
    a = np.dot(np.transpose(matrix), matrix)
    eigenVal = np.array(eigenValue(a))
    eigenVal[(eigenVal < 0)] = 0
    eigenVal = eigenVal[(eigenVal != 0)]

    # get singular values from eigen values
    singVal = (np.diag(np.sqrt(eigenVal)))
    return singVal

def svdCompress(matrix, k):
    u = getU(matrix)
    print("u done") #nnati apus
    v = getVTranpose(matrix)
    print("v done") #nanti apus
    s = getSigma(matrix)
    print("s done") #nanti apus
    return (u[:, :k], s[:k, :k], v[:k, :])

#for testing
def svdResult(matrix):
    u = getU(matrix)
    v = getVTranpose(matrix)
    sigma = getSigma(matrix)
    svd = np.dot(u, np.dot(sigma, v))
    return svd

def compress(percentage): #add img as param later
    # just for testing -- delete this part later
    currDir = os.path.dirname(__file__)
    path = os.path.join(currDir, './static/images/jvv_absen.jpg') # testing aja nnt apush
    tes = cv2.imread(path)
    img = np.array(tes)
    cv2.imshow('BEFORE', img)
    cv2.waitKey()
    print(img.shape)
    # delete until here

    # initialize rgb
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # make scale out of percentage
    if (len(img) < len(img[0])):
        k = round(percentage / 100 * len(img))
    else:
        k = round(percentage / 100 * len(img[0]))

    # buat bandingin aja ini kalo pake linalg bawaan
    '''
    ur, sr, vr = np.linalg.svd(r)
    ug, sg, vg = np.linalg.svd(g)
    ub, sb, vb = np.linalg.svd(b)
    ur, sr, vr = (ur[:, :k], np.diag(sr[:k]), vr[:k, :])
    ug, sg, vg = (ug[:, :k], np.diag(sg[:k]), vg[:k, :])
    ub, sb, vb = (ub[:, :k], np.diag(sb[:k]), vb[:k, :])
    '''
    # ini yang kita :V
    ur, sr, vr = svdCompress(r, k)
    print("\nred dapet\n")
    ug, sg, vg = svdCompress(g, k)
    print("\ngreen dapet\n")
    ub, sb, vb = svdCompress(b, k)
    print("\nblue dapet\n")
    
    # ini nanti apus
    print(ur.shape)
    print(sr.shape)
    print(vr.shape) 

    # make compressed r, g, b
    rScaled = np.dot(ur, np.dot(sr, vr))
    gScaled = np.dot(ug, np.dot(sg, vg))
    bScaled = np.dot(ub, np.dot(sb, vb))

    # insert compressed r, g, b to matrix
    imgScaled = np.zeros(img.shape)
    imgScaled[:, :, 0] = bScaled
    imgScaled[:, :, 1] = gScaled
    imgScaled[:, :, 2] = rScaled

    # check for values outside of range of RGB (0-255)
    imgScaled[imgScaled > 255] = 255
    imgScaled[imgScaled < 0] = 0

    imgScaled = imgScaled.astype(float) / 255
    cv2.imshow("AFTER", imgScaled)
    cv2.waitKey()

# testing implementation
matrix = [[3,1,1], [-1,3,1]]
mit = [[5, 5], [-1, 7]]
'''
tes = np.dot(np.transpose(mit), mit)

eigval, eigvec = np.linalg.eig(tes)
print("matriks\n", tes)
print("eigval\n", eigval)
print("eigval\n", eigvec)

u, s, vt = np.linalg.svd(matrix)
print("matriks u:\n", u)
print("matriks sigma:\n", s)
print("matriks vt:\n", vt)
'''
compress(2)