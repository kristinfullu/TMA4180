import numpy as np    
import scipy
from problem import problem, E_elast_cable, grad_cable, E_ext, E_elast_bars, E_grav_bars,  grad_ext, grad_bar_elast, grad_bar_grav, penalty_func, grad_penalty_func


X0 = np.array([ # Initial structure
    [1,1,1],
    [-1,1,1],
    [-1,-1,1],
    [1,-1,1],
    [0.5,0.5,10],
    [-0.5,0.5,10],
    [-0.5,-0.5,10],
    [0.5,-0.5,10]])

# Initialise edges, lenghts, element_type
edges = np.array([[0,1],[0,3],[0,4],[0,7],[1,2],[1,4],[1,5],[2,3],[2,5],[2,6],[3,6],[3,7],[4,5],[4,7],[5,6],[6,7]])
L_bars = np.array([2,2,10,8,2,8,10,2,8,10,8,10,1,1,1,1])
element_type = np.array([0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0])

n = len(X0)                          # Number of nodes
num_edges = len(edges)               # Number of edges

g = 9.81
Mass = np.zeros(n)                      # Mass of each of the free nodes
H0 = np.identity(len(X0)*3) #*0.7       # Approximation of "hessian" 


def modified_free(problem, X):
    return E_elast_cable(problem,X)+E_ext(problem,X) + E_elast_bars(problem,X) + E_grav_bars(problem,X) + penalty_func(problem, X)

def modified_free_grad(problem, X):
 
    return grad_cable(problem,X) + grad_ext(problem,X) + grad_bar_elast(problem,X) + grad_bar_grav(problem,X) + grad_penalty_func(problem, X)


# Initialise test problem:
free_prob = problem(modified_free, modified_free_grad, edges , L_bars, Mass, X0, H0 , element_type, k=0.1, c=1 , rho=(10**-4)/g,alpha=5,factor=0.1)
free_prob2 = problem(modified_free, modified_free_grad, edges , L_bars, Mass, X0, H0 , element_type, k=0.1, c=1 , rho=(10**-4)/g,alpha=5,factor=0.01)



