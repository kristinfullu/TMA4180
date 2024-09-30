import numpy as np
import scipy
import scipy.linalg

def StrongWolfe(problem, X,initial_value, grad_x, p, initial_descent, 
                initial_step_length = 1.0, 
                c1 = 1e-3, 
                c2 = 0.9, 
                max_extrapolation_iterations = 100,
                max_interpolation_iterations = 50,
                rho = 2.0):
    '''
    Implementation of a bisection based bracketing method
    for the strong Wolfe conditions
    This code is based on the code provided in the course TMA4180 Optimization by Markus Grasmair
    -------------------------------------------------------------------------------
    Input:
    problem                      - class instance of test problem
    X                            - intial X
    inital_value                 - initial evaluation of funtion
    grad_x                       - initial gradient
    p                            - initial search diection
    initial_descent              - intial descent innerprod(p,grad_x)
    initial_step_length          - Initial step length
    c1                           - constant for Armijo Rule
    c2                           - constant for Curvature Condition
    max_extrapolation_iterations - max number of extrapolation iterations
    max_interpolation_iterations - max number of interpolation iterations
    rho                          - make upper bound alphaR larger by a factor rho

    Output:
    next_x                       - the next X,  X + alpha*d
    next_value                   - function evaluated at next X
    next_grad                    - gradient evaluated at next X
    tot_func_evals               - Total number of function evaluations
    alphaR/alpha                 - returns steplength alphaR/alpha satisfying our conditions
    '''


    # initialise the bounds of the bracketing interval
    alphaR = initial_step_length
    alphaL = 0.0

    # Armijo condition and the two parts of the Wolfe condition
    # are implemented as Boolean variables
    next_x = X+alphaR*p.reshape((len(problem.X0),3))
    next_value = problem.function(problem, next_x)
    tot_func_evals = 1
    next_grad = problem.gradient(problem, next_x)
    Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
    descentR = np.inner(p,next_grad)
    curvatureLow = (descentR >= c2*initial_descent)
    curvatureHigh = (descentR <= -c2*initial_descent)


    # Check whether Armijo holds and curvatureLow fails.
    itnr = 0
    while (itnr < max_extrapolation_iterations and (Armijo and (not curvatureLow))):
        itnr += 1
        # alphaR is a new lower bound for the step length
        # the old upper bound alphaR needs to be replaced with a larger step length
        alphaL = alphaR
        alphaR *= rho
        # update function value and gradient
        next_x = X+alphaR*p.reshape((len(problem.X0),3))
        next_value = problem.function(problem, next_x)
        tot_func_evals += 1
        next_grad = problem.gradient(problem,next_x)
        # update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)

    # at that point we should have a situation where alphaL is too small
    # and alphaR is either satisfactory or too large
    if(Armijo and curvatureLow and curvatureHigh):
        return next_x,next_value,next_grad,tot_func_evals,alphaR
    alpha = np.copy(alphaR)
    itnr = 0
    
    # Use bisection in order to find a step length alpha that satisfies all conditions
    while (itnr < max_interpolation_iterations and (not (Armijo and curvatureLow and curvatureHigh))):
        itnr += 1
        if (Armijo and (not curvatureLow)):
            # the step length alpha was still too small
            # replace the former lower bound with alpha
            alphaL = alpha
        else:
            # the step length alpha was too large
            # replace the upper bound with alpha
            alphaR = alpha
        # choose a new step length as the mean of the new bounds
        alpha = (alphaL+alphaR)/2
        # update function value and gradient
        next_x = X+alpha*p.reshape((len(problem.X0),3))
        next_value = problem.function(problem,next_x)
        tot_func_evals += 1
        next_grad = problem.gradient(problem,next_x)
        # update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alpha*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
    
    # return the next iterate as well as the function value and gradient there
    # (in order to save time in the outer iteration; we have had to do these
    # computations anyway)
    if(itnr == max_interpolation_iterations):
       print("Step length not converged")
    return next_x,next_value,next_grad,tot_func_evals,alpha
 
def BFGS(problem, max_steps = 50, err = 1e-6,convergence_plot = False ): 
    '''
    BFGS algorithm
    This code is based on the code provided in the course TMA4180 Optimization by Markus Grasmair
    -----------------------------------
    Input:
    problem     - class
    max_steps   - max iterations
    err         - stops if the differnence in new-old approx hessian is as small as err

    Output:
    X           - solution X that minimises problem.function
    '''
    # Initialise the variable x
    X = problem.X0
    # Initialise the quasi-Newton matrix as the identity
    H = np.identity(np.size(X))
    
    if convergence_plot:
        if problem.minimisers is None:
            print("A convergence plot requires the (analytic) minimisers of the function to be set.")
            convergence_plot = False
            errors = False
        else:
            errors = []
    
    
    n_step = 0
    alpha = problem.alpha
    current_value = problem.function(problem, X)
    current_gradient = problem.gradient(problem, X)
    func_evals = 1
    norm_grad = np.linalg.norm(current_gradient)
    

    # Main loop 
    while ((n_step < max_steps) and (norm_grad > err)):
        n_step += 1
        p = -H@current_gradient
        descent = np.inner(p,current_gradient)
        X_old = X
        gradient_old = np.copy(current_gradient)

        X,current_value,current_gradient,func_ls,_ = StrongWolfe(problem, X,current_value, current_gradient,p, descent)
        func_evals += func_ls
        norm_grad = np.linalg.norm(current_gradient)

        if convergence_plot:
            errors = errors + [np.linalg.norm(X-problem.minimisers)]

        s = X - X_old
        y = current_gradient - gradient_old
        rho = 1/np.inner(y,s.flatten())
        if n_step==1:
            H = H*(1/(rho*np.inner(y,y)))
        z = H.dot(y)
        H += -rho*(np.outer(s,z) + np.outer(z,s)) + rho*(rho*np.inner(y,z)+1)*np.outer(s,s)
            
    if n_step < max_steps:
        print("Converged after {} iterations.".format(n_step))
    else:
        print("Did not converge after {} iterations.".format(max_steps))
    print("A total of {} function and gradient evaluations were required.".format(func_evals))

    return X, errors