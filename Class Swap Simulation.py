import numpy as np  
import sympy as sy 
import matplotlib.pyplot as plt

# This program simulates the class swap birth process with parameters
# s12, s21, L1, L2 and n jumps. At each jump, one of 4 actions is possible --
# class 1 swaps to class 2, class 1 speciates, class 2 swaps to class 1,
# or class 2 speciates. After experiencing n jumps, the process ends and returns
# the final number of species in class 1 and class 2 as well as the times at
# which each jump occurred.

def classswapsim(s12, s21, L1, L2, n):
    # Initializes N1, the array counting the species in class 1 at each jump,
    # and N2, the array counting the species in class 2 at each jump. Also initializes
    # T, the array of jump times.
    N1 = np.zeros(n)
    N1[0] = 1
    N2 = np.zeros(n)
    N2[0] = 0
    T = np.zeros(n)
    # Gives names to each of the 4 possible actions a jump can undertake.
    jumpactions = ['swap2', 'spec1', 'swap1', 'spec2']
    # Experiences n jumps, each jump as an iteration of the for loop.
    for i in range(0,n-1):
        # Generates a wait time for the jump from the wait time distribution,
        # and then updates the jump time.
        dT = -(np.log(1 - np.random.rand())) / (N1[i] * (s12 + L1) + N2[i] * (s21 + L2))
        T[i + 1] = T[i] + dT
        # Updates the probabilities of each of the 4 actions.
        pswap2 = (N1[i] * s12) / (N1[i] * (s12 + L1) + N2[i] * (s21 + L2))
        pspec1 = (N1[i] * L1) / (N1[i] * (s12 + L1) + N2[i] * (s21 + L2))
        pswap1 = (N2[i] * s21) / (N1[i] * (s12 + L1) + N2[i] * (s21 + L2))
        pspec2 = (N2[i] * L2) / (N1[i] * (s12 + L1) + N2[i] * (s21 + L2))
        # Picks one of the four actions with the updated distribution.
        action = np.random.choice(jumpactions, 1, p = [pswap2, pspec1, pswap1, pspec2])
        # If action is swap2, one member of class 1 becomes a member of class 2.
        if action == ['swap2']:
            N1[i + 1] = N1[i] - 1
            N2[i + 1] = N2[i] + 1
        # If action is spec1, class 1 adds a member and class 2 remains the same.
        if action == ['spec1']:
            N1[i + 1] = N1[i] + 1
            N2[i + 1] = N2[i]
        # If action is swap1, one member of class 2 becomes a member of class 1.
        if action == ['swap1']:
            N1[i + 1] = N1[i] + 1
            N2[i + 1] = N2[i] - 1
        # If action is spec2, class 2 adds a member and class 1 remains the same.
        if action == ['spec2']:
            N1[i + 1] = N1[i]
            N2[i + 1] = N2[i] + 1
    return T, N1, N2

# Simulates k times the class swap birth process with
# parameters s12, s21, L1, L2 and n jumps, keeping track of
# each of the final T, N1, N2 arrays from each iteration of the simulation.
# In the first plot, it plots all the simulated N1, N2's with the supposed
# line (defined by y) it appears to be becoming parallel to. 
# In the second plot, it plots all the simulated T, N1 arrays together.
# In the third plot, it plots all the simulated T, N2 arrays together.
# Finally, in the fourth plot it plots all the simulated T, N1/(N1 + N2)
# arrays together along with the ratio it should converge to (defined by
 # z).

def repeatedsimulator(s12, s21, L1, L2, n, k):
    # N1upperbound just keeps track of the biggest value in each N1 array
    # so that when the limiting line is plotted, it is plotted to scale with
    # the other data.
    N1upperbound = 0
    # Tupperbound keeps track of the biggest value in each T array
    # so that when the limiting ratio is plotted, it is plotted to scale with
    # the other data.
    Tupperbound = 0
    # Sets up the plot figure. Also labels axis and titles the plots.
    # Also adds a box with parameter values.
    fig = plt.figure(figsize=(30, 30))
    ax1, ax2, ax3, ax4 = fig.subplots(4, 1)
    ax1.set_title('N2 vs N1', fontsize = 20)
    ax1.set(xlabel = 'N1', ylabel = 'N2')
    ax2.set_title('N1 vs t', fontsize = 20)
    ax2.set(xlabel = 't', ylabel = 'N1')
    ax2.text(0.01, 0.8,
             f"""
             $\lambda_{{1}} = {L1}$ \n\
             $s_{{12}} = {s12}$ \n\
             $\lambda_{{2}} = {L2}$ \n\
             $s_{{21}} = {s21}$
             """, 
             bbox = dict(facecolor = 'white', alpha = 0.5),
             fontsize = 14,
             transform = ax2.transAxes)
    ax3.set_title('N2 vs t', fontsize = 20)
    ax3.set(xlabel = 't', ylabel = 'N2')
    ax3.text(0.01, 0.8,
             f"""
             $\lambda_{{1}} = {L1}$ \n\
             $s_{{12}} = {s12}$ \n\
             $\lambda_{{2}} = {L2}$ \n\
             $s_{{21}} = {s21}$
             """, 
             bbox = dict(facecolor = 'white', alpha = 0.5),
             fontsize = 14,
             transform = ax3.transAxes)
    ax4.set_title('R1 vs t', fontsize = 20)
    ax4.set(xlabel = 't', ylabel = 'R1')
    ax4.text(0.91, 0.04,
             f"""
             $\lambda_{{1}} = {L1}$ \n\
             $s_{{12}} = {s12}$ \n\
             $\lambda_{{2}} = {L2}$ \n\
             $s_{{21}} = {s21}$
             """, 
             bbox = dict(facecolor = 'white', alpha = 0.5),
             fontsize = 14,
             transform = ax4.transAxes)
    # Simulates k times.
    for j in range(0, k):
        [T, N1, N2] = classswapsim(s12, s21, L1, L2, n)
        if np.amax(N1) >= N1upperbound:
            N1upperbound = np.amax(N1)
        if np.amax(T) >= Tupperbound:
            Tupperbound = np.amax(T)
        ax1.plot(N1, N2, color = (1 - j/k, 0, j/k))
        ax2.step(T, N1, color = (1 - j/k, 0, j/k), where = 'post')
        ax3.step(T, N2, color = (1 - j/k, 0, j/k), where = 'post')
        ax4.step(T, N1/(N1 + N2), color = (1 - j/k, 0, j/k), where = 'post')
    # Defines Q, since it gets used a bit from here down.
    Q = (-(s12 + s21 - L1 - L2) - np.sqrt((s12 + s21 + L1 - L2)**2 - 4 * s12 * (L1 - L2))) / 2
    # Adds the convergent line plot to ax1.
    r = s12 / (L1 - s12 - Q)
    x1 = np.linspace(0, N1upperbound, 100)
    y = r * x1 - r
    ax1.plot(x1, y, color = 'lime')
    # Adds the convergent ratio plot to ax4 -- this looks a little weird because
    # plotting a constant function is a little silly, but that is all
    # that is going on here.
    x2 = np.linspace(0, Tupperbound, 100)
    z = np.full((100, 1), (L1 - s12 - Q) / (L1 - Q))
    ax4.plot(x2, z, color = 'lime')
    plt.show()
    
repeatedsimulator(0.8, 5, 1.2, 8, 10**4, 50)