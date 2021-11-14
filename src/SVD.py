from Eigen import simultaneous_power_iteration
from PIL import Image
import numpy as np
import cv2

def svd(matrix, k):
    """ Singular Value Decomposition
    Decomposes a matrix into U, Sigma, and VTranspose.

    :param matrix: A matrix to be processed
    :param k: A scale for which the matrix size will be minimized
    
    :return: Three minimized matrices of U, Sigma, and VTranspose 
    """
    # Find Eigen Vals and Eigen Vecs of AT.A
    a = np.dot(np.transpose(matrix), matrix) 
    eigVal, eigVec = simultaneous_power_iteration(a, k) 
    
    # Find Singular Val (sqrt of non-zero absolute Eigen Vals)
    singval = [] 
    for i in eigVal:
        singval.append(np.sqrt(np.abs(i))) 
    singval = np.array(singval)
    
    # Sort Eigen Val & Corresponding Eigen Vecs (descending)
    idx = singval.argsort()[::-1]
    singval = singval[idx]
    eigVec = eigVec[:, idx]
    singval = singval[singval != 0.0]

    # Find V, VTranspose (k x n), and Sigma (k x k)
    v = eigVec
    vt = (np.transpose(v))[:k, :]
    sigma = (np.diag(singval))[:k, :k] 

    # Find U (m x k)
    u = np.dot(matrix, v[:, :k])
    for i in range(k): 
        u[:, i] = u[:, i] / singval[i] 

    return u, sigma, vt

def compress(img, percentage):
    """ Compression Function
    Compresses an image using SVD with scale k depending on compression percentage.

    :param img: An image to be processed in the form of 3D array
    :param percentage: Compression percentage
    
    :return: Compressed image
    """
    # Convert image values to float32
    img = img.astype(np.float32)

    # Extract RGB/RGBA matrices
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    try: a = img[:, :, 3]
    except: pass
    
    # Make scale k out of percentage
    m = img.shape[0]
    n = img.shape[1]
    percent = percentage / 100
    k = round((m * n) * percent / (m + n + 1))

    # Scale RGB/RGBA matrices using SVD
    try: 
        ub, sb, vb = svd(b, k)
        bScaled = np.dot(ub, np.dot(sb, vb))
    except: bScaled = b
    try: 
        ug, sg, vg = svd(g, k)
        gScaled = np.dot(ug, np.dot(sg, vg))
    except: gScaled = g
    try: 
        ur, sr, vr = svd(r, k)
        rScaled = np.dot(ur, np.dot(sr, vr))
    except: rScaled = r

    try: 
        ua, sa, va = svd(a, k)
        aScaled = np.dot(ua, np.dot(sa, va))
    except: pass

    # Make L image out of scaled RGB/RGBA matrices
    redImg = (Image.fromarray(rScaled)).convert("L")
    greenImg = (Image.fromarray(gScaled)).convert("L")
    blueImg = (Image.fromarray(bScaled)).convert("L")
    try: 
        alphaImg = (Image.fromarray(aScaled)).convert("L")
        # Merge RGBA images into a single image
        imgScaled = Image.merge("RGBA" ,(redImg,greenImg,blueImg, alphaImg))
        # Convert from RGBA to BGRA (PIL to OpenCV)
        opencvimg = cv2.cvtColor(np.array(imgScaled), cv2.COLOR_RGBA2BGRA)
    except: 
        # Merge RGB images into a single image
        imgScaled = Image.merge("RGB" ,(redImg,greenImg,blueImg))
        # Convert from RGB to BGR (PIL to OpenCV)
        opencvimg = cv2.cvtColor(np.array(imgScaled), cv2.COLOR_RGB2BGR)

    return (opencvimg)