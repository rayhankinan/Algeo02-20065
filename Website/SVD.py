import os # ini nanti ilangin cm buat testing
import cv2
import numpy as np
from Eigen import simultaneous_power_iteration
import math

def svd(matrix, k):
    a = np.dot(np.transpose(matrix), matrix) # get A trans * A
    eigVal, eigVec = simultaneous_power_iteration(a, k) #<-- ini kode kita, tp utk testing skrg pake library dulu
    #eigVal, eigVec = np.linalg.eig(a) # find eig val and eig vec of A trans * A
    
    singval = [] 
    for i in eigVal:
        singval.append(np.sqrt(np.abs(i))) # get singular values from eig val (if sing val is negative make it absolute value)
    singval = np.array(singval)
    
    # bagian di bawah ini apus aja kalo udah gapake library eigen
    idx = singval.argsort()[::-1] # sort eigen value decreasing
    singval = singval[idx]
    eigVec = eigVec[:, idx]
    singval = singval[singval != 0.0]
    v = eigVec
    
    vt = (np.transpose(v))[:k, :]
    sigma = (np.diag(singval))[:k, :k] # get sigma (diagnolized matrix with singular value)
    u = np.dot(matrix, v[:, :k])

    for i in range(k): # dividing ui (column) with sigma[i]
        u[:, i] = u[:, i] / singval[i] # get ui = A * vi / singular value i (v is sliced according to scale k, so that we get a sliced u matrix as well)

    # for testing
    #print(u.shape)
    #print(sigma.shape)
    #print(vt.shape)

    return u, sigma, vt

def compress(percentage): #add img as param later
    # just for testing -- delete this part later
    currDir = os.path.dirname(__file__)
    path = os.path.join(currDir, './static/images/jvv absen.jpg') # testing aja nnt apush
    tes = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    cv2.imshow("yey", tes)
    cv2.waitKey()
    img = tes.astype(np.float32)
    print(img.shape)
    # delete until here

    # initialize rgb
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    try: a = img[:, :, 3]
    except: pass
    
    # make scale out of percentage
    if (img.shape[0] < img.shape[1]):
        k = round(percentage / 100 * len(img))
    else:
        k = round(percentage / 100 * len(img[0]))

    # k = KALO MAU K MANUAL INPUT DISINI
    # tes punya kita yang paling baru :V
    ub, sb, vb = svd(b, k)
    ug, sg, vg = svd(g, k)
    ur, sr, vr = svd(r, k)
    try: ua, sa, va = svd(a, k)
    except: pass
    #print(ub.shape)
    #print(sb.shape)
    #print(vb.shape)

    # make compressed r, g, b
    rScaled = np.dot(ur, np.dot(sr, vr))
    gScaled = np.dot(ug, np.dot(sg, vg))
    bScaled = np.dot(ub, np.dot(sb, vb))
    try: aScaled = np.dot(ua, np.dot(sa, va))
    except: pass

    # insert compressed r, g, b to matrix
    try: imgScaled = cv2.merge((bScaled, gScaled, rScaled, aScaled))
    except: imgScaled = cv2.merge((bScaled, gScaled, rScaled))
    #imgScaled[:, :, 3] = aScaled

    # check for values outside of range of RGB (0-255)
    imgScaled[imgScaled > 255] = 255
    imgScaled[imgScaled < 0] = 0
    #cv2.imwrite("yayoyo.jpg", imgScaled)
    return (imgScaled)
    #imgScaled = imgScaled.astype(float) / 255
    #cv2.imshow("INI PAKE SVD KITA BOUSZ", imgScaled)
    #cv2.waitKey()
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
    return imgScaled
    #imgScaled = imgScaled.astype(float) / 255
    #cv2.imshow("INI PAKE SVD LIBRARY BOUSZ", imgScaled)
    #cv2.waitKey()
'''
compress(3)
'''
# ini buat ngetes SVD in general aja
a = [[1.02650, 0.92840, 0.54947, 0.98317, 0.71226, 0.55847], [0.92889, 0.89021, 0.49605, 0.93776, 0.62066, 0.52473], [0.56184, 0.49148, 0.80378, 0.68346, 1.02731, 0.64579], [0.98074, 0.93973, 0.69170, 1.03432, 0.87043, 0.66371], [0.69890, 0.62694, 1.02294, 0.87822, 1.29713, 0.82905], [0.56636, 0.51884, 0.65096, 0.66109, 0.82531, 0.55098], [1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6]] #ganti jadi matrix apapun itu
print("A matrix")
u, s, v = svd(a, 5)
print("RESULT:\n")
print("U matrix")
print(u)
print("S matrix")
print(s)
print("V matrix")
print(v)
print("End matrix")
print(np.dot(u, np.dot(s, v)))

print(np.dot(np.transpose(u), u))
print(np.dot(v, np.transpose(v)))
'''