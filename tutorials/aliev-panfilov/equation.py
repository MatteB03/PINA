# this is a file given to me by Pierfrancesco that calculates the analytical solution
# of the problem, we use this in the tutorial to compare our solution with the expected one 

import numpy as np
import matplotlib.pyplot as plt

def laplacian(V, h):
    """
    Calculates the laplacian of a 1D array V using central FD.
    """
    d2V = np.zeros_like(V)
    d2V[1:-1] = (V[:-2] - 2 * V[1:-1] + V[2:]) / (h ** 2)
    return d2V

def AlPan(y, Istim):
    """
    Calculate the time derivative for Aliev-Panfilov equations.
    
    Parameters:
      y: (X,2) shape array whose columns are V and W respectively.
      Istim: external stimulus array (size: X).
      
    Return:
      dydt: (X,2) shape array containing [dV/dt, dW/dt].
    """
    # model parameters
    a = 0.01
    k = 8.0
    mu1 = 0.2
    mu2 = 0.3
    epsi = 0.002
    b = 0.15
    h_cell = 0.1  # cell length (space units)
    D = 0.1       # diffusion coefficient

    V = y[:, 0]
    W = y[:, 1]
    
    # V laplacian (diffusion)
    dV_diff = 4 * D * laplacian(V, h_cell)
    
    # W equation
    dWdt = (epsi + mu1 * W / (mu2 + V)) * (-W - k * V * (V - b - 1))
    
    # V (nonlinear, diffusion and stimulus)
    dVdt = -k * V * (V - a) * (V - 1) - W * V + dV_diff + Istim
    
    dydt = np.column_stack((dVdt, dWdt))
    return dydt


def AlievPanfilov1D_RK_Istim():
    """
    Simulates the Aliev-Panfilov model in 1D using Runge-Kutta 4 (RK4) method.
    
    Parameters:
      BCL      : Basic Cycle Length (AU), time elapsed between stimuli.
      ncyc     : No. of cycles.
      extra    : Additional time after BCL * ncyc (AU).
      ncells   : No. of internal cells (length of the 1D cable).
      iscyclic : False for the cable (Neumann boundary conditions), True for the ring (periodic conditions, not fully implemented here).
      flagmovie: True to show real-time visualization.
      
    Returns:
      Vsav, Wsav: Array with Vsav, Wsav saved solutions to plot
    """

    # parameters we used to compare the solution with the result of our PINN
    BCL = 70
    ncyc = 1
    extra = 0
    ncells = 70
    iscyclic = False
    flagmovie = True
    # here we expand the domain to include boundary cells for bound_conds
    X = ncells + 2  
    
    # here we define the geometry of our stimulus: stimulate the first 5 internal cells
    stimgeo = np.zeros(X, dtype=bool)
    stimgeo[:5] = True
    
    # time parameters
    dt = 0.005  # time step
    gathert = int(round(1 / dt))  # save every 1 AU
    tend = BCL * ncyc + extra  # overall simulation lenght (AU)
    stimdur = 1  # stimulus duration (AU)
    
    # defining the stimulus: when stimulated, Ia = 0.1; 0 elsewhen
    Ia = 0.1 * stimgeo.astype(float)
    
    # initial conditions: V and W are initially = 0.01 on all the domain 
    V = np.full(X, 0.01)
    W = np.full(X, 0.01)
    
    # preallocate array to save the solutions on the internal cells
    n_save = int(np.ceil(tend / dt / gathert))
    Vsav = np.zeros((ncells, n_save))
    Wsav = np.zeros((ncells, n_save))
    
    ind = 0   # iterations counter
    kk = 0    # counter for the number of stimuli
    Istim = np.zeros(X)
    # Initialize the state y (each row contains [V, W])
    y = np.column_stack((V, W))

    # Setup for the movie if flagmovie is True
    #if flagmovie:
    #    plt.ion()
    #    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    t_values = np.arange(dt, tend + dt, dt)
    sav_index = 0

    # Time loop with RK4 integration
    for t in t_values:
        ind += 1
        # Apply the stimulus if we are at the appropriate time
        if t >= BCL * kk and kk < ncyc:
            Istim = Ia.copy()
        # After stimdur, stop the stimulus and increment the counter
        if t >= BCL * kk + stimdur * 2:
            kk += 1
            Istim = np.zeros(X)
    
        # Compute the RK4 coefficients
        k1 = AlPan(y, Istim)
        k2 = AlPan(y + dt / 2 * k1, Istim)
        k3 = AlPan(y + dt / 2 * k2, Istim)
        k4 = AlPan(y + dt * k3, Istim)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
        # Extract V and W from the updated array
        V = y[:, 0]
        W = y[:, 1]
    
        # Apply boundary conditions: Neumann conditions for V (1D cable)
        if not iscyclic:
            V[0] = V[1]
            V[-1] = V[-2]
        else:
            # For a ring, periodic conditions should be implemented here
            pass
    
        # Save data every gathert iterations
        if ind % gathert == 0:
            Vsav[:, sav_index] = V[1:-1]  # exclude boundary cells
            Wsav[:, sav_index] = W[1:-1]
            sav_index += 1
        
        # If flagmovie is True, show a movie of the propagation
        #if flagmovie:
        #    ax1.clear()
        #    V_display = np.tile(V[1:-1], (max(ncells // 20, 1), 1))
        #    im1 = ax1.imshow(V_display, vmin=0, vmax=1, aspect='auto', cmap='jet')
        #    ax1.set_title(f'V (AU) - Time: {t:.0f} AU')
        #    ax1.set_xlabel('x (cells)')
        #    fig.colorbar(im1, ax=ax1)
            
        #    ax2.clear()
        #    W_display = np.tile(W[1:-1], (max(ncells // 20, 1), 1))
        #    im2 = ax2.imshow(W_display, vmin=0, vmax=1, aspect='auto', cmap='jet')
        #    ax2.set_title(f'W (AU) - Time: {t:.0f} AU')
        #    ax2.set_xlabel('x (cells)')
        #    fig.colorbar(im2, ax=ax2)
            
        #    plt.pause(0.01)

    #if flagmovie:
    #    plt.ioff()
    #    plt.show()

    # Final plot of the saved solutions
    #plt.figure(figsize=(10, 8))

    #plt.subplot(2, 1, 1)
    #plt.title('V (AU)')
    #plt.imshow(Vsav.T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='jet')
    #plt.ylabel('Time (AU)')
    #plt.xlabel('x (cells)')
    #plt.colorbar()

    #plt.subplot(2, 1, 2)
    #plt.title('W (AU)')
    #plt.imshow(Wsav.T, aspect='auto', origin='lower', vmin=0, vmax=2.4, cmap='jet')
    #plt.ylabel('Time (AU)')
    #plt.xlabel('x (cells)')
    #plt.colorbar()

    #plt.show()
    #plt.savefig("./plot_finale.png")

    return Vsav, Wsav