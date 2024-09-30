# Define a class to make implementation easier
import numpy as np
import scipy

class problem(object):
    def __init__(self, function, gradient, edges ,L, M, X0, H0, element, fixed_nodes = None ,k=0, c=0, rho=0, minimisers = None,alpha=0, factor=0):
        self.function = function       # Energy function
        self.gradient = gradient       # Gradient of energy function

        self.edges = edges             # Edges (i-j)
        self.L = L                     # Array of lenghts,  l_ij
        self.M = M                     # Array of masses, mi

        self.fixed_nodes = fixed_nodes # Fixed nodes
        self.X0 = X0                   # Initial guess free nodes
        self.H0 = H0                   # Initial guess, approx of "hessian"
        self.k = k                     # Material parameter cables
        self.c = c                     # Material parameter bars
        self.rho = rho                 # Line density of bars
        self.element = element         # Element type, cable=0 or bar=1
        self.minimisers = minimisers   # Points that minimise function
        self.alpha = alpha             # Penalty parameter
        self.factor = factor           # Factor for function


# Functions to calculate energy of structure
def E_elast_cable(problem, X):
    ''' 
    Elastic energy of cables
    -------------------------------------------------------
    Input:
    problem       - Class with necessary constants/variables
    X             - (n,3)-array with n points in 3 dimensions

    Output:
    E_elast_cable - Elastic energy of cables
    ''' 
    E_elast_cable = 0
    k = problem.k
    i = 0
    if problem.fixed_nodes is None:
        nodes = X

    else:
        nodes = np.concatenate((problem.fixed_nodes, X))

    for e in problem.edges:
        if problem.element[i] == 0: # The edge is a cable
            
            diff = scipy.linalg.norm(nodes[e[0]]-nodes[e[1]])
            l = problem.L[i]
            if diff> l:
                E_elast_cable+= k/(2*l**2)*(diff-l)**2
                
        i+=1
    return E_elast_cable

def E_ext(problem, X):
    '''
    External energy of structure
    ----------------------------
    Input:
    problem - Class with necessary constants/variables
    X       - (n,3)-dim array with n points in 3 dimensions

    Output:
    E_ext   - External energy of structure
    '''
    g = 9.81
    E_ext = np.sum(g*problem.M*X[::,2])
    return E_ext

def E_elast_bars(problem, X):
    '''
    Elastic energy of bars
    -----------------------
    Input:
    problem      - Class with necessary constants/variables
    X            - (n,3)-dim array with n points in 3 dimensions

    Output:
    E_elast_bars - Elastic energy of bars
    '''
    E_elast_bars = 0
    c = problem.c
    i = 0
    if problem.fixed_nodes is None:
        nodes = X
    else:
        nodes = np.concatenate((problem.fixed_nodes, X))

    for e in problem.edges:
        if problem.element[i] == 1: # The edge is a bar
            diff = scipy.linalg.norm(nodes[e[0]]-nodes[e[1]])
            l = problem.L[i]
            E_elast_bars+= c/(2*l**2)*(diff-l)**2
        i+=1
    return E_elast_bars

def E_grav_bars(problem, X):
    '''
    gravitational potential energy due to mass of the bars
    ---------------------------------------------------
    Input:
    problem      - Class with necessary constants/variables
    X            - (n,3)-dim array with n points in 3 dimensions

    Output:
    E_grav_bars - gravitational potential energy of bars
    '''
    g = 9.81
    rho = problem.rho
    E_grav_bars = 0
    i = 0
    if problem.fixed_nodes is None:
        nodes = X
    else:
        nodes = np.concatenate((problem.fixed_nodes, X))

    for e in problem.edges:
        if problem.element[i] == 1: # The edge is a bar
            l = problem.L[i]
            E_grav_bars+= rho*g*l/2*(nodes[e[0]][2]+nodes[e[1]][2])
        i+=1

    return E_grav_bars


# Gradients of the energy functions
def grad_cable(problem, X):

    '''
    Gradient of elastic energy of cables
    -------------------------------------
    Input:
    problem      - Class with necessary constants/variables
    X            - (n,3)-dim array with n points in 3 dimensions

    Output:
    grad         - (n*3)-dim array, gradient of E_elast_cable
    '''
    n = len(X)
    grad = np.zeros(n*3)

    k = problem.k
    g = 9.81

    if problem.fixed_nodes is None:
        nodes = X
        n_fixed = 0

    else:
        nodes = np.concatenate((problem.fixed_nodes, X))
        n_fixed = len(problem.fixed_nodes)

    it = 0
    for e in problem.edges:
        if problem.element[it] == 0: # The edge is a cable
            if e[0]<n_fixed: # These are fixed nodes
                # print("e, fixed nodes: ", e)
                diff = scipy.linalg.norm(nodes[e[0]]-nodes[e[1]])
                l = problem.L[it]
                if diff> l:
                    start =(e[1]-n_fixed)*3     # Index til punkt x_j i gradienten
                    grad[start:start+3] += k/(l**2)*(diff-l)*(nodes[e[1]]-nodes[e[0]])/diff 
                    
            else: # These are free nodes
                # print("e, free nodes: ", e)
                diff = scipy.linalg.norm(nodes[e[0]]-nodes[e[1]])
                l = problem.L[it]
                if diff > l:
                    start_i = (e[0]-n_fixed)*3       # Index for derivert mhp punkt "x_i" i gradienten
                    start_j = (e[1]-n_fixed)*3       # Index for derivert mhp punkt "x_j" i gradienten
                    grad[start_i:start_i+3] += k/(l**2)*(diff-l)*(nodes[e[0]]-nodes[e[1]])/diff
                    grad[start_j:start_j+3] += k/(l**2)*(diff-l)*(nodes[e[1]]-nodes[e[0]])/diff
                    
        it+=1    
    
    return grad

def grad_ext(problem,X):
    '''
    Gradient of external energy
    -------------------------------------
    Input:
    problem      - Class with necessary constants/variables
    X            - (n,3)-dim array with n points in 3 dimensions

    Output:
    grad         - (n*3)-dim array, gradient of E_ext
    '''
    g = 9.81 
    n = len(X) 
    grad = np.zeros(n*3)
    grad[2::3] += problem.M*g # External force, z-component)
    return grad

def grad_bar_elast(problem, X):

    '''
    Gradient of elastic energy in bars
    -----------------------------------
    Input:
    problem      - Class with necessary constants/variables
    X            - (n,3)-dim array with n points in 3 dimensions

    Output:
    grad         - (n*3)-dim array, gradient of E_bar_elast
    '''
    n = len(X)  # Number of free nodes
    grad = np.zeros(n*3) # Gradient of E_elast_bars wrt. points X

    c = problem.c   # Constant 
    

    if problem.fixed_nodes is None:
        nodes = X
        n_fixed = 0
    else:   
        nodes = np.concatenate((problem.fixed_nodes, X)) # All nodes, fixed and free
        n_fixed = len(problem.fixed_nodes) # Number of fixed nodes
    it = 0
    for e in problem.edges:
        if problem.element[it] == 1: # The edge is a bar
            if e[0]<n_fixed: # These are fixed nodes
                diff = scipy.linalg.norm(nodes[e[0]]-nodes[e[1]])
                l = problem.L[it]
                start =(e[1]-n_fixed)*3     # Index til punkt x_j i gradienten
                grad[start:start+3] += c/(l**2)*(diff-l)*(nodes[e[1]]-nodes[e[0]])/diff
            else: # These are free nodes
                diff = scipy.linalg.norm(nodes[e[0]]-nodes[e[1]])
                l = problem.L[it]
                start_i = (e[0]-n_fixed)*3       # Index for derivert mhp punkt "x_i" i gradienten
                start_j = (e[1]-n_fixed)*3       # Index for derivert mhp punkt "x_j" i gradienten
                grad[start_i:start_i+3] += c/(l**2)*(diff-l)*(nodes[e[0]]-nodes[e[1]])/diff
                grad[start_j:start_j+3] += c/(l**2)*(diff-l)*(nodes[e[1]]-nodes[e[0]])/diff
        it+= 1
    
    return grad

def grad_bar_grav(problem, X):

    '''
    Gradient of gravitational potential energy
    -------------------------------------
    problem      - Class with necessary constants/variables
    X            - (n,3)-dim array with n points in 3 dimensions

    Output:
    grad         - (n*3)-dim array, gradient of E_bar_grav
    '''
    n = len(X)
    grad = np.zeros(n*3)

    k = problem.k
    rho = problem.rho
    L = problem.L

    g = 9.81
    if problem.fixed_nodes is None:
        n_fixed = 0
    else:
        n_fixed = len(problem.fixed_nodes)

    it = 0

    for e in problem.edges:
        if problem.element[it] == 1: # The edge is a bar
            if e[0]<n_fixed: # These are fixed nodes
                start =(e[1]-n_fixed)*3
                grad[start+2] += rho*g*L[it]/2
            else: # These are free nodes
                start = (e[0]-n_fixed)*3
                grad[start+2] += rho*g*L[it]/2
                start = (e[1]-n_fixed)*3
                grad[start+2] += rho*g*L[it]/2
        it+= 1
        
    return grad


def f(x,factor): 
    # Input is one point
    return factor*(x[0]**2+x[1]**2)

def g_func(x,factor):
    # Input is one point
    return f(x,factor)- x[2]

def g_func_deriv(x,factor):
    # Input is one point
    return np.array([factor*2*x[0],factor*2*x[1],-1])

def penalty_func(problem,X):
    value = 0
    for x in X:
        if g_func(x,problem.factor) >= 0:
            value += problem.alpha*g_func(x,problem.factor)**2
    return value

def grad_penalty_func(problem,X):

    n = len(X) # Number of free nodes
    grad = np.zeros(n*3) # Gradient of penalty function wrt. points X, flattened
    alpha = problem.alpha
    it = 0

    for x in X:
        if g_func(x,problem.factor) >= 0:
            # print(grad.shape)
            grad[it*3:it*3+3] = 2*alpha*g_func(x,problem.factor)*g_func_deriv(x,problem.factor) 
        it += 1

    return grad
        
