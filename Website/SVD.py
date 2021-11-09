import os # ini nanti ilangin cm buat testing
import cv2
import numpy as np
from Eigen import eigenValue, eigenVectorNorm

def svd(matrix, k):
    a = np.dot(np.transpose(matrix), matrix) # get A trans * A
    #eigVal, eigVec = eigenValue(a), eigenVectorNorm(a) <-- ini kode kita, tp utk testing skrg pake library dulu
    eigVal, eigVec = np.linalg.eigh(a) # find eig val and eig vec of A trans * A

    singval = [] 
    for i in eigVal:
        singval.append(np.sqrt(np.abs(i))) # get singular values from eig val (if sing val is negative make it absolute value)
    singval = np.array(singval) 

    # bagian di bawah ini apus aja kalo udah gapake library eigen
    idx = singval.argsort()[::-1] # sort eigen value decreasing
    singval = singval[idx]
    vt = eigVec[:, idx] # sort eigen vec with corresponding eigen val, assign it as V Transpose (each eigen vector as row --> V Transpose)

    sigma = np.diag(singval) # get sigma (diagnolized matrix with singular value)

    v = np.transpose(vt) # get v (each eigen vector as col --> V)
    u = np.dot(matrix, v[:, :k]) # get ui = A * vi (v is sliced according to scale k, so that we get a sliced u matrix as well)

    for i in range(k): # dividing ui (column) with sigma[i]
        u[:, :i] / singval[i]
        
    sigma = sigma[:k, :k] # slice sigma according to scale k
    vt = vt[:k, :] # slice vt according to scale k

    # for testing
    print(u.shape)
    print(vt.shape)
    print(sigma.shape)

    '''
    THIS IS PART OF THE DEPRECATED SOURCE CODE (getU, getSigma, getVTranspose)
    u = np.array(getU(matrix))
    sigma, vt = np.array(getSigmaVT(matrix))

    u = u[:, :k]
    sigma = sigma[:k, :k]
    vt = vt[:k, :]
    '''
    return u, sigma, vt

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
    if (img.shape[0] < img.shape[1]):
        k = round(percentage / 100 * len(img))
    else:
        k = round(percentage / 100 * len(img[0]))

    # tes punya kita yang paling baru :V
    ub, sb, vb = svd(b, k)
    ug, sg, vg = svd(g, k)
    ur, sr, vr = svd(r, k)

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
    '''
# uncomment ini kalo mo bandingin jawaban pake svd linalg library

    ur, sr, vr = np.linalg.svd(r)
    ug, sg, vg = np.linalg.svd(g)
    ub, sb, vb = np.linalg.svd(b)
    ur, sr, vr = (ur[:, :k], np.diag(sr[:k]), vr[:k, :])
    ug, sg, vg = (ug[:, :k], np.diag(sg[:k]), vg[:k, :])
    ub, sb, vb = (ub[:, :k], np.diag(sb[:k]), vb[:k, :])
    
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

# THESE ARE DEPRECATED
def getU(matrix): # left singular value
    # get eigen values and eigen vectors of a matrix
    a = np.dot(matrix, np.transpose(matrix))
    eigval, eigvec = np.linalg.eig(a)
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    u = np.transpose(eigvec) # ini hasilnya uda dinormalize
    return u

def getSigmaVT(matrix): # right singular value
    # get eigen values and eigen vectors of a matrix
    a = np.dot(np.transpose(matrix), matrix)
    eigval, eigvec = np.linalg.eig(a) # ini hasilnya uda dinormalize
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    for i in eigval:
        if (i < 0):
            i = 0
        i = np.sqrt(i)
    sigma = np.diag(eigval)
    return sigma, eigvec

    a = np.dot(np.transpose(matrix), matrix)
    eigenVal = np.array(eigenValue(a))
    eigenVal[(eigenVal < 0)] = 0
    eigenVal = eigenVal[(eigenVal != 0)]

    # get singular values from eigen values
    singVal = (np.diag(np.sqrt(eigenVal)))
    return singVal
'''
compress(80)