import matplotlib.pyplot as plt
import numpy as np

def fu(x, y,factor):
    return (x**2 + y**2)*factor

def plot_initial_and_sol(test, test_sol, errors, title):
    '''
    Plots initial structure and solution as 3D plots next to each other, along with a semilogy plot of errors.
    '''

    fig = plt.figure(figsize=(18, 6))  # Adjusted for better fit of three plots
    fig.suptitle(title, fontsize=30)
    gs = fig.add_gridspec(1, 4, width_ratios=[2, 2, 2, 3])  # Adjusted grid spec for a smaller semilogy plot

    # ----------------------- Plot initial structure ------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    nodes_initial = np.concatenate((test.fixed_nodes, test.X0))
    
    # Plot fixed and free nodes
    ax1.scatter3D(test.fixed_nodes[:, 0], test.fixed_nodes[:, 1], test.fixed_nodes[:, 2], color="green", s=100, label="Fixed Nodes")
    ax1.scatter3D(test.X0[:, 0], test.X0[:, 1], test.X0[:, 2], color="blue", s=100, label="Free Nodes")
    
    # Plot edges
    for i, e in enumerate(test.edges):
        points = np.array([nodes_initial[e[0]], nodes_initial[e[1]]])
        color, linestyle = ("red", "-") if test.element[i] == 1 else ("black", "--")
        ax1.plot3D(*points.T, color=color, linestyle=linestyle)
    
    ax1.set_title("Initial Structure", fontsize=20)
    ax1.legend()
    # ax1.view_init(elev=90, azim=0)  # Viewing angle from above

    # --------------- Plot solution ------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    nodes_solved = np.concatenate((test.fixed_nodes, test_sol))
    
    ax2.scatter3D(test.fixed_nodes[:, 0], test.fixed_nodes[:, 1], test.fixed_nodes[:, 2], color="green", s=100)
    ax2.scatter3D(test_sol[:, 0], test_sol[:, 1], test_sol[:, 2], color="blue", s=100)
    
    # Plot edges
    for i, e in enumerate(test.edges):
        points = np.array([nodes_solved[e[0]], nodes_solved[e[1]]])
        color, linestyle = ("red", "-") if test.element[i] == 1 else ("black", "--")
        ax2.plot3D(*points.T, color=color, linestyle=linestyle)
    
    ax2.set_title("Solution", fontsize=20)
    # ax2.view_init(elev=90, azim=0)  # Viewing angle from above

    # --------------- Plot solution, view from top ------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    nodes_solved = np.concatenate((test.fixed_nodes, test_sol))
    
    ax3.scatter3D(test.fixed_nodes[:, 0], test.fixed_nodes[:, 1], test.fixed_nodes[:, 2], color="green", s=100)
    ax3.scatter3D(test_sol[:, 0], test_sol[:, 1], test_sol[:, 2], color="blue", s=100)
    
    # Plot edges
    for i, e in enumerate(test.edges):
        points = np.array([nodes_solved[e[0]], nodes_solved[e[1]]])
        color, linestyle = ("red", "-") if test.element[i] == 1 else ("black", "--")
        ax3.plot3D(*points.T, color=color, linestyle=linestyle)
    
    ax3.set_title("Solution, seen from above", fontsize=20)
    ax3.view_init(elev=90, azim=0)  # Viewing angle from above

    

    # ------------------ Semilogy plot of errors -----------------------------
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.semilogy(errors, marker='o', linestyle='-', color='green')
    ax4.set_title("Logarithmic Error Plot", fontsize=20)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Error")
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f'{title}.png')

    plt.show()

def plot_free(test,test_sol,title,factor):
    '''
    Plots initial structure and solution as 3d-plot next to each other.
    '''
 
    # ----------------------- Plot initial structure ------------------------
    fig = plt.figure(figsize=(10, 7))                                
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    nodes = test.X0
    
    ax1.scatter3D(nodes[:,0],nodes[:,1],nodes[:,2],color = "blue", s = 200)                              # Free nodes - blue
    i = 0
    for e in test.edges:
        point = np.concatenate((nodes[e[0]], nodes[e[1]]))
        if test.element[i] == 1:
            ax1.plot3D(point[::3], point[1::3], point[2::3], color = "red")                                    # Bars - red line
        else:
            ax1.plot3D(point[::3], point[1::3], point[2::3], color = "black",linestyle="dashed")               # Cables - black, dashed line
        i +=1

    ax1.set_title("Initial structure", fontsize = 20)     # Title
    # ax1.view_init(elev=90, azim=0)         # View from "above"

    # --------------- Plot solution ------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    # Plot nodes
    nodes_solved = test_sol                                        # Solved structure

    ax2.scatter3D(test_sol[:,0],test_sol[:,1],test_sol[:,2],color = "blue", s = 200)                            # Free nodes - blue

    # Plot edges
    i = 0
    for e in test.edges:
        point = np.concatenate((nodes_solved[e[0]], nodes_solved[e[1]]))
        if test.element[i] == 1:
            ax2.plot3D(point[::3], point[1::3], point[2::3], color = "red")                                    # Bars - red line
        else:
            ax2.plot3D(point[::3], point[1::3], point[2::3], color = "black", linestyle="dashed")              # Cables - black, dashed line
        i +=1

    x = np.linspace(-100*factor, 100*factor, 100)
    y = np.linspace(-100*factor, 100*factor, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function
    Z = fu(X, Y,factor)

    ax2.scatter3D(X.flatten(), Y.flatten(), Z.flatten(), cmap='viridis', alpha=0.01)


    ax2.set_title("Solution", fontsize = 20)          # Title
    # ax2.view_init(elev=90, azim=0)    # View from "above"

    # --------------- Plot solution from above ------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    # Plot nodes
    nodes_solved = test_sol                                        # Solved structure

    ax3.scatter3D(test_sol[:,0],test_sol[:,1],test_sol[:,2],color = "blue", s = 200)                            # Free nodes - blue

    # Plot edges
    i = 0
    for e in test.edges:
        point = np.concatenate((nodes_solved[e[0]], nodes_solved[e[1]]))
        if test.element[i] == 1:
            ax3.plot3D(point[::3], point[1::3], point[2::3], color = "red")                                    # Bars - red line
        else:
            ax3.plot3D(point[::3], point[1::3], point[2::3], color = "black", linestyle="dashed")              # Cables - black, dashed line
        i +=1

    x = np.linspace(-100*factor, 100*factor, 100)
    y = np.linspace(-100*factor, 100*factor, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function
    Z = fu(X, Y,factor)

    ax3.scatter3D(X.flatten(), Y.flatten(), Z.flatten(), cmap='viridis', alpha=0.01)


    ax3.set_title("Solution, seen from above", fontsize = 20)          # Title
    ax3.view_init(elev=90, azim=0)    # View from "above"

    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()
    

