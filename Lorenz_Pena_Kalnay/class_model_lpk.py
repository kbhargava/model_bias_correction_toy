import numpy as nnnp
from scipy.integrate import odeint

#===============================================================================
# DEFINE FUNCTIONS USED BY THE CLASS
#===============================================================================

#-------------------------------------------------------------------------------
def dxdt(x, t, params, params_cpl=[none]*5,F= np.zeros(3)):
#-------------------------------------------------------------------------------
    # x is the state vector of len 3 in uncoupled case and 6 in coupled case
    # F represents the paramterized forcing used in case of the uncoupled system
    # params are the parameters sigma, rho (r), beta (b)
    # params_cpl are the coupling  parameters tau, k1, S,c, cz 

    sigma,r,b  = params
    tau, k1, S,c, cz = params_cpl
    dn = len(x)
    if dn == 3:
        dx[0] = sigma*(x[1]- [0])        - F[0]
        dx[1] = r*x[0] -x[1] - x[0]*x[2] + F[1]
        dx[2] = x[0]*x[1]    - b*x[2]    + F[3]
    elif dn == 6 :
        dx[0] = sigma*(x[1]- [0])                   - c *(S*x[3]+k1)
        dx[1] = r*x[0] -x[1] - x[0]*x[2]            + c *(S*x[4]+k1) 
        dx[2] = x[0]*x[1]    - b*x[2]               + cz*x[5]
        dx[3] = tau*sgma(x[4]*x[3]                  - c *(x[0]+k1)
        dx[4] = tau*r*x[3] -tau*x[4]-tau*S*X[3]*X[5]+ c *(x[1]+k1)
        dx[5] = tau*S*x[3]*x[4] - tau*b*x[5]        - cz*x[2]   
    else:
        print ("Supported state vector lengths are 3 for the uncoupled atmosphere and 6 for coupled system")
        print ("You supplied a state vector of length {}".format(dn))
        print ("returning None")
        dx    = None
    return dx

#-------------------------------------------------------------------------------
def Ja(x, t, params, params_cpl=[None]*5):
#-------------------------------------------------------------------------------
    # Computes the analytical Jacobian
    dn = len (x)
    sigma,r,b  = params
    tau, k1, S,c, cz = params_cpl
    J = np.eye(dn)
    if dn == 3:
        J = [[-sigma, sigma,    0  ],
             [r-x[2],    -1, -x[0] ],
             [  x[1],  x[0],    -b ]]
    elif dn == 6 :
    
        J = [[-sigma, sigma,   0,              -c*S,         0,           0 ],
             [r-x[2],    -1,  -x[0],              0,    c*x[3],           0 ],
             [  x[1],  x[0],  -b,                 0,         0, c         z ],
             [    -c,     0,   0,        -tau*sigma, tau*sigma,           0 ],
             [     0,     c,   0, tau*r -tau*S*x[5],      -tau, -tau*S*x[3] ],
             [     0,     0, -cz,        tau*S*x[5], tau*S*[3],      -tau*b ]] 
    else:
        print ("Supported state vector lengths are 3 for the uncoupled atmosphere and 6 for coupled system")
        print ("You supplied a state vector of length {}".format(dn))
        print ("returning identity matrix")
        dx    = None
    J = np.matrix(J)
    return J


#===============================================================================
class lorenzlpk:
#===============================================================================
    
    #---------------------------------------------------------------------------
    # Initialize model parameters
    #---------------------------------------------------------------------------
    def __init__(self,params=[10.0,28,8.0/3.0,], params_cpl=[0.1,-11,1,1,1], F=np.zeros(3)):
        self.params     = params
        self.params_cpl = params_cpl
        self.F          = F
    
    #---------------------------------------------------------------------------
    # Run model
    #---------------------------------------------------------------------------
    def run(self,state0,t):
        states = 

