import os # ini nanti ilangin cm buat testing
import cv2
import numpy as np
from Eigen import eigenValue, eigenVectorNorm

def svd(matrix, k):
    a = np.dot(np.transpose(matrix), matrix) # get A trans * A
    #eigVal, eigVec = eigenValue(a), eigenVectorNorm(a) <-- ini kode kita, tp utk testing skrg pake library dulu
    eigVal, eigVec = eigenValue(a), eigenVectorNorm(a) # find eig val and eig vec of A trans * A
    print("finish eigen value and vector")
    singval = [] 
    for i in eigVal:
        singval.append(np.sqrt(np.abs(i))) # get singular values from eig val (if sing val is negative make it absolute value)
    singval = np.array(singval) 

    # bagian di bawah ini apus aja kalo udah gapake library eigen
    idx = singval.argsort()[::-1] # sort eigen value decreasing
    singval = singval[idx]
    eigVec = eigVec[:, idx] # sort eigen vec with corresponding eigen val
    
    vt = np.transpose(eigVec)
    sigma = np.diag(singval) # get sigma (diagnolized matrix with singular value)
    v = np.transpose(vt) # get v
    u = np.zeros(shape=(len(matrix), k))

    for i in range(k): # dividing ui (column) with sigma[i]
        u[:, i] = np.dot(matrix, eigVec[:, i]) / singval[i] # get ui = A * vi / singular value i (v is sliced according to scale k, so that we get a sliced u matrix as well)

    u = np.array(u)
        
    sigma = sigma[:k, :k] # slice sigma according to scale k
    vt = vt[:k, :] # slice vt according to scale k

    # for testing
    print(u.shape)
    print(sigma.shape)
    print(vt.shape)

    return u, sigma, vt

def compress(percentage): #add img as param later
    # just for testing -- delete this part later
    currDir = os.path.dirname(__file__)
    path = os.path.join(currDir, './static/images/bjir.jpg') # testing aja nnt apush
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
        b, g, r = b.transpose(), g.transpose(), r. transpose()
    else:
        k = round(percentage / 100 * len(img[0]))
    
    # tes punya kita yang paling baru :V
    ub, sb, vb = svd(b, k)
    ug, sg, vg = svd(g, k)
    ur, sr, vr = svd(r, k)
    print(ub.shape)
    print(sb.shape)
    print(vb.shape)

    # make compressed r, g, b
    rScaled = np.dot(ur, np.dot(sr, vr))
    gScaled = np.dot(ug, np.dot(sg, vg))
    bScaled = np.dot(ub, np.dot(sb, vb))

    if (img.shape[0] < img.shape[1]):
        bScaled, gScaled, rScaled = bScaled.transpose(), gScaled.transpose(), rScaled.transpose()

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
'''
compress(100)

# ini buat ngetes SVD in general aja
#a = [[3, 2, 3, 7, 8, 9], [2, 3, 4, 6, 8, 10]] #ganti jadi matrix apapun itu
#u, s, v = svd(a, 2)
#print("RESULT:\n")
#print(u)
#print(s)
#print(v)
#print(np.dot(u, np.dot(s, v))
