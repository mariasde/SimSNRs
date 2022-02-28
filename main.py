# This script simulates SNRs in the Galaxy as per Arias 2022

import numpy as np
from functions import run_simulation, plot_results, confusion_plots


if __name__ == '__main__':
    
    # find the detection fraction of SNRs
    iters = [5, 10, 20]
    ns = np.arange(250,3001,100)
    results = run_simulation(iters,ns)
    plot_results(iters,results) # saves results in Figs
    
    # find the confused fraction
    iterations = 10
    ns = np.arange(250,2001,100)
    confusion_plots(iterations,ns)