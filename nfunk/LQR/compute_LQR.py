'''
File in order to compute an optimal LQR controller
x[k] = [theta , thetaDot]
'''

import numpy as np
import scipy.linalg

# Discrete-time LQR
def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # ref Bertsekas, p.151
 
    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    
    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))
    
    eigVals, eigVecs = scipy.linalg.eig(A - B * K)
    
    return K

# Define Params:
g = 10.0
m = 1.0
l = 1.0
dt = 0.05

A = np.matrix([[1,dt],[dt*(3*g)/(2*l),1]])
B = np.matrix([[0],[dt*(3)/(m*l**2)]])

Q = np.diag([1,0.1])
R = np.diag([0.1])

# Calculate Gain Matrix:
K = dlqr(A,B,Q,R)

print ("Gain Matrix is:")
print (K)

# PREVIOUSLY USED LQR Matrix: <-> SO FAR ALREADY CONSISTENT WITH ENV (1.0,0.1,0.001)
#Q = np.diag([10,1])
#R = 0.01
#K = [[19.80210901  6.03515527]]


