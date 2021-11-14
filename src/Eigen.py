import numpy as np
import time

def simultaneous_power_iteration(A, k):
    '''
    Algorithm to find eigenValues and eigenVectors of a matrix using 
    simultaneous power iteration.

    k is how many eigenVector the user wants to find (in this case, how many pixels being trimmed)
    '''

    n, m = A.shape 
    Q = np.random.rand(n, k) #Make a random n x k matrix
    Q, _ = np.linalg.qr(Q) #Use QR decomposition to Q

 
    for i in range(10):
        Z = A.dot(Q)
        Q, R = np.linalg.qr(Z)
    #Do the same thing over and over until it converges
    return np.diag(R), Q
