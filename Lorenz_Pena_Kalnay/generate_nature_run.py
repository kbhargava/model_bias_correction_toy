import numpy as np

from class_model_lpk import lorenzlpk
from class_state_vector import state_vector

#-------------------------------------------------------------------------------
# Setup initial state
#-------------------------------------------------------------------------------


xdim = 6
F = np.zeros(3)
params = [10.0,28,8.0/3.0]   # sigma, rho, b
params_cpl = [0.1,-11,1,1,1] # tau, k1, S, c, cz
name = 'x_nature'

t0 = 0.0
tf = 20
dt = 0.001  

state0 = np.ones(xdim) + np.random.uniform(low=-1.0, high=1.0, size = (xdim))
tvec = np.arange(t0, tf, dt)

#------------------------------------------------------------------
# (Optional) Update the initial starting point
#------------------------------------------------------------------
state0 = [ -5.35168657,  -6.75463366,  23.32140275,  13.18101471,
           -10.82812224, -3.05244954]
#state0 = [1,1,0,-1,0,1]
#------------------------------------------------------------------
# (Optional) Add initial perturbation
#------------------------------------------------------------------
# From previous run:
#Climatological Mean:
#[2.37742045 2.26356092 2.45364065 2.31241994 2.02554909]
#Climatological Standard Deviation:
#[3.63477096 3.63919927 3.30788215 3.76514026 3.62503822]

#name = 'x_freerun'
#initial_perturbation = np.squeeze(0.01*(np.random.rand(1,3)*2-1))
#print('initial_perturbation = ', initial_perturbation)
#climate_std =  [3.63477096 3.63919927 3.30788215 3.76514026 3.62503822]
#print('climate_std = ', climate_std)
#state0 = state0 + initial_perturbation*climate_std
#print('initial state = ', state0)

#------------------------------------------------------------------
# Setup state vector object
#------------------------------------------------------------------
sv = state_vector(params=params,params_cpl= params_cpl,x0=state0,t=tvec,name=name)

#------------------------------------------------------------------
# Initialize the l96 object
#------------------------------------------------------------------
lpk = lorenzlpk(params=params, params_cpl= params_cpl,F=F)

#------------------------------------------------------------------
# Run L96 to generate a nature run with the specified parameters
#------------------------------------------------------------------
print('Run lorenz pena and Kalnay model...')
trajectory = lpk.run(sv.x0,sv.t)
sv.setTrajectory(trajectory)

#------------------------------------------------------------------
# Output the beginning and end states, and compute model climatology
#------------------------------------------------------------------
print(trajectory[:,-1])

lpk.plot_anim(trajectory,tvec)
#lpk.plot_3d(trajectory,tvec)
#------------------------------------------------------------------
# Store the nature run data
#------------------------------------------------------------------
#outfile = name+'.pkl'
#sv.save(outfile)
