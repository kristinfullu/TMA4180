import numpy as np    
import scipy
from problem import problem, E_elast_cable, grad_cable, E_ext,  grad_ext


# Initialise free nodes:
X=np.array([
    [1,1,1/2],
    [-10,20,1/2],
    [-1,-1,22],
    [1,5,1/2]])

# Initialise fixed nodes:
fixed_nodes =np.array([
    [5,5,0],
    [-5,5,0],
    [-5,-5,0],
    [5,-5,0]]) 

# Edges, lenghts
edges = np.array([[0,4],[1,5],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]) # Fixed nodes frist, then free nodes (indexing)

n_fixed = len(fixed_nodes)           # Number of fixed nodes
n_X = len(X)                         # Number of free nodes
n_nodes = n_X+n_fixed                # Total number of nodes
num_edges = len(edges)               # Number of edges

# Elements, lenghts
elements = np.zeros(num_edges)       # 0 = cable, 1 = bar
L_test = np.ones(num_edges)*3        # Lenght of cable between all (i,j), corresponding to vector of edges

g = 9.81                             # Gravity
M = np.ones(n_X)*1/(6*g)             # Mass of each of the free nodes
H0 = np.identity(len(X)*3)*0.7       # Initial approximation of hessian

# Energy function:
def cable_nets(problem, X):
    return E_elast_cable(problem,X)+E_ext(problem,X)

# Gradient:
def cable_net_grad(problem, X):
    return grad_cable(problem,X) + grad_ext(problem,X)

# Eaxct solution:
exact = np.array([
    [2,2,-3/2],
    [-2,2,-3/2],
    [-2,-2,-3/2],
    [2,-2,-3/2]])

# Initialise test problem:
cable_nets_prob = problem(cable_nets, cable_net_grad,edges , L_test, M, X,H0 ,elements, fixed_nodes, k=3, minimisers=exact )


