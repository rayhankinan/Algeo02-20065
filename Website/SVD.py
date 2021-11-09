import os # ini nanti ilangin cm buat testing
import cv2
import numpy as np
from Eigen import eigenValue, eigenVectorNorm

def svd(matrix, k):
    a = np.dot(np.transpose(matrix), matrix)
    eigVal, vt = eigenValue(a), eigenVectorNorm(a)
    #eigVal, vt = np.linalg.eig(a)

    singval = []
    for i in eigVal:
        if (i < 0):
            i = 0
        singval.append(np.sqrt(i))
    singval = np.array(singval)
    sigma = np.diag(singval)

    sigma = sigma[:k, :k]
    vt = vt[:k, :]

    v = np.transpose(vt)
    u = np.dot(matrix, v[:, :k])
    for i in range(k):
        u[:, :i] * singval[i]
    print(u.shape)
    print(vt.shape)
    print(sigma.shape)

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
    if (len(img) < len(img[0])):
        k = round(percentage / 100 * len(img))
    else:
        k = round(percentage / 100 * len(img[0]))

    # tes punya kita :V
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
    # buat bandingin aja ini kalo pake linalg bawaan

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
    u = np.array(eigenVectorNorm(a)) # ini hasilnya uda dinormalize
    return u

def getVTranpose(matrix): # right singular value
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
'''
compress(30)