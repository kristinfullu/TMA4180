import numpy as np    
import scipy
from problem import problem, E_elast_cable, grad_cable, E_ext, E_elast_bars, E_grav_bars,  grad_ext, grad_bar_elast, grad_bar_grav



#Initilise free nodes:
X0=np.array([
    [-0.5,0,4],
    [0,1,10],
    [1,0,12],
    [0.5,0.5,11]])

# Initialise fixed nodes:
fixed_nodes =np.array([
    [1,1,0],
    [-1,1,0],
    [-1,-1,0],
    [1,-1,0]]) 

# Initialise edges, lenghts, element_type
edges = np.array([[0,4],[0,7],[1,4],[1,5],[2,5],[2,6],[3,6],[3,7],[4,5],[4,7],[5,6],[6,7]])
L_bars = np.array([10,8,8,10,8,10,8,10,1,1,1,1])
element_type = np.array([1,0,0,1,0,1,0,1,0,0,0,0])


n_fixed = len(fixed_nodes)           # Number of fixed nodes
n_X = len(X0)                        # Number of free nodes
n_nodes = n_X+n_fixed                # Total number of nodes
num_edges = len(edges)               # Number of edges

g = 9.81
Mass = np.zeros(n_X)                    # Mass of each of the free nodes
H0 = np.identity(len(X0)*3) #*0.7       # Approximation of "hessian" 


# Initialise energy function:
def bars_e(problem, X):
    return E_elast_cable(problem,X)+E_ext(problem,X) + E_elast_bars(problem,X) + E_grav_bars(problem,X)

# Initialise gradient:
def bars_grad(problem, X):
    return grad_cable(problem,X) + grad_ext(problem,X) + grad_bar_elast(problem,X) + grad_bar_grav(problem,X)

# Exact solution
s = 0.70970
t = 9.54287

exact = np.array([
    [-s,0,t],
    [0,-s,t],
    [s,0,t],
    [0,s,t]])




# Initialise test problem:
bars_prob = problem(bars_e, bars_grad, edges , L_bars, Mass, X0, H0 , element_type, fixed_nodes, k=0.1, c=1 , rho=0, minimisers=exact)

# Second initial values
X02=np.array([
    [-0.5,0,-4],
    [0,1,-10],
    [1,0,-12],
    [0.5,0.5,-11]])

bars_prob2 = problem(bars_e, bars_grad, edges , L_bars, Mass, X02, H0 , element_type, fixed_nodes, k=0.1, c=1 , rho=0, minimisers=None)




