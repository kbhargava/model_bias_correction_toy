import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#===============================================================================
# DEFINE FUNCTIONS USED BY THE CLASS
#===============================================================================

#-------------------------------------------------------------------------------
def dxdt(t,x,params, params_cpl=[None]*5,F= np.zeros(3)):
#-------------------------------------------------------------------------------
    # x is the state vector of len 3 in uncoupled case and 6 in coupled case
    # F represents the paramterized forcing used in case of the uncoupled system
    # params are the parameters sigma, rho (r), beta (b)
    # params_cpl are the coupling  parameters tau, k1, S,c, cz 

    sigma,r,b  = params
    tau, k1, S,c, cz = params_cpl
    dn = len(x)
    dx = np.zeros(dn)
    if dn == 3:
        dx[0] = sigma*(x[1]- x[0])        - F[0]
        dx[1] = r*x[0] -x[1] - x[0]*x[2] + F[1]
        dx[2] = x[0]*x[1]    - b*x[2]    + F[3]
    elif dn == 6 :
        dx[0] = sigma*(x[1]- x[0])                  - c *(S*x[3]+k1)
        dx[1] = r*x[0] -x[1] - x[0]*x[2]            + c *(S*x[4]+k1) 
        dx[2] = x[0]*x[1]    - b*x[2]               + cz*x[5]
        dx[3] = tau*sigma*(x[4]-x[3])               - c *(x[0]+k1)
        dx[4] = tau*r*x[3] -tau*x[4]-tau*S*x[3]*x[5]+ c *(x[1]+k1)
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
             [  x[1],  x[0],  -b,                 0,         0,          cz ],
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
    def run(self,state0,t_list):
        #states = 
        states = solve_ivp(dxdt, (t_list[0],t_list[-1]), state0, method = "RK45", dense_output=False,
                    args=(self.params, self.params_cpl,self.F),t_eval=t_list)
        return states.y

    #---------------------------------------------------------------------------
    # Compute approximate TLM with I + Df(x0)*dt
    #---------------------------------------------------------------------------
    def compute_TLMa(self, states):
        nr = np.shape(states.y)
        nc = np.shape(states.t)
        I  = np.eye(nc)

        # Compute linear propogator for each timestep
        maxit = len(states.t)
        Mhist= []
        for i in range (maxit):
            if (i< maxit-1):
                dt = t[i+1] -t[i]
            else:
                dt = t[-1] -t[-2]

            # Evaluate Jacobian 
            Df = Ja(states.y[i],states.t[i],self.params,self.params_cpl)
                    
            # Compute approximate linear propogator
            M = I + Df*dt
            Mhist.append(deepcopy(M))
        return Mhist
    
    #---------------------------------------------------------------------------
    # Plot animation
    #---------------------------------------------------------------------------
    def plot_anim(self, states, cvec, figname = "lorenz_pena_kalnay-3d",
                            plot_title = 'Lorenz Pena Kanay attractor'):
        nr,nc = np.shape(states)
        # Set up figure and 3D axis for animation
        fig = plt.figure()
        ax1  = fig.add_subplot(1,2,1, projection = "3d" )
        ax2  = fig.add_subplot(1,2,2, projection = "3d" )
        #ax1.axis('off')

        # Choose color for each trajectory
        colors = plt.cm.jet(np.linspace(0,1,nc))

        # Set up line and points
        line1, = ax1.plot([],[],[],'-',c='g', label="Atmosphere")
        pt1,   = ax1.plot([],[],[],'o', c='r')
        line2, = ax2.plot([],[],[],'-',c='b', label="Ocean") 
        pt2,   = ax2.plot([],[],[],'o', c='r')
        #lines  = [line1, line2]
        #pts    = [pt1, pt2]
        #lines  = np.array([line1, line2]).flatten()
        #pts    = np.array([pt1, pt2]).flatten()

        # prepare the axes limits 
        # write this part based on the limits you see when running
        ax1.set_xlim((-25,25))
        ax1.set_ylim((-35,35))
        ax1.set_zlim((0,55))
        ax2.set_xlim((-50,50))
        ax2.set_ylim((-150,150))
        ax2.set_zlim((-150,150))
        # set point of view: specified by (altitude degrees, azimuth degrees)
        ax1.view_init(30,0)
        ax2.view_init(45,0)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend ()
        ax2.legend()
        
        #plt.legend()
        # initializatoin function: plot te background for each frame
        def init():
            line1.set_data([],[])
            line1.set_3d_properties([])
            pt1.set_data([],[])
            pt1.set_3d_properties([])
            #for line, pt in zip(lines,pts):
            #    line.set_data([],[])
            #    line.set_3d_properties([])
            #    pt.set_data([],[])
            #    pt.set_3d_properties([])
            return line1,pt1

        # animate function. This will be calles sequentially with the frame number
        def animate(i):
            i = (2*i)%np.shape(states)[1]

            x  = states[0,:i]
            y  = states[1,:i]
            z  = states[2,:i]
            x2 = states[3,:i]
            y2 = states[4,:i]
            z2 = states[5,:i]
            
            line1.set_data(x,y)
            line1.set_3d_properties(z)
            pt1.set_data(x[-1:],y[-1:])
            pt1.set_3d_properties(z[-1:])

            line2.set_data(x2,y2)
            line2.set_3d_properties(z2)
            pt2.set_data(x2[-1:],y2[-1:])
            pt2.set_3d_properties(z2[-1:])
            
            """
            lines[0].set_data(x,y)
            lines[0].set_3d_properties(z)
            pts[0].set_data(x[-1:],y[-1:])
            pts[0].set_3d_properties(z[-1:])
            
            lines[1].set_data(x2,y2)
            lines[1].set_3d_properties(z2)
            pts[1].set_data(x2[-1:],y2[-1:])
            pts[1].set_3d_properties(z2[-1:])
            """
            ax1.view_init(30,0.3*i)
            ax2.view_init(30,0.3*i)
            fig.canvas.draw()
            fig.suptitle("Timesteps = %s" % (i/1000))
            return line1,pt1
        # instantiate the animator
        anim = animation.FuncAnimation(fig,animate,init_func=init,frames=5000, interval =1, blit = True)
        anim.save('{0}.mp4'.format(figname),fps=30, extra_args=['-vcodec', 'libx264'])
        #plt.show()
    
    #---------------------------------------------------------------------------
    # Plot 3d fig
    #---------------------------------------------------------------------------
    def plot_3d(self, states, cvec, figname = "lorenz_pena_kalnay-3d",
                            plot_title = 'Lorenz Pena Kanay attractor'):
        nr,nc = np.shape(states)
        # Set up figure and 3D axis for animation
        fig = plt.figure()
        ax1  = fig.add_subplot(1,2,1, projection = "3d" )
        ax2  = fig.add_subplot(1,2,2, projection = "3d" )
            
        x  = states[0,:]
        y  = states[1,:]
        z  = states[2,:]
        x2 = states[3,:]
        y2 = states[4,:]
        z2 = states[5,:]

        # Set up line and points
        ax1.plot(x,y,z,'-',c='g', label="Atmosphere")
        ax2.plot(x2,y2,z2,'-',c='b', label="Ocean") 

        # prepare the axes limits 
        # write this part based on the limits you see when running
        """
        ax1.set_xlim((0,425))
        ax1.set_ylim((-20,30))
        ax1.set_zlim((0,55))
        ax2.set_xlim((-5,25))
        ax2.set_ylim((-170,0))
        ax2.set_zlim((-50,25))
        #ax1.view_init(30,0)
        #ax2.view_init(45,0)
        """
        ax1.legend ()
        ax2.legend()
        plt.show()
