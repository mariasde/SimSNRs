import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist 


def simulate_D(n):
    ## returns n diameters in pc according to the
    # distribution of SNR diameters in M33 --Long+10
    
    long_diams = [123,29,100,39,45,56,73,51,39,93,20,52,33,46,28,51,33,30,71,51,67,27,25,99,25,73,44,179,66,76,20,21,67,32,19,18,32,51,13,55,97,43,85,29,30,39,50,21,43,71,60,33,34,45,44,23,36,67,42,14,60,73,54,48,50,59,45,109,47,21,20,32,57,31,44,21,24,16,58,8,34,56,31,32,43,13,48,60,92,42,36,101,20,20,23,18,14,67,51,26,58,39,48,39,50,66,33,77,25,54,51,84,38,22,47,42,66,53,33,43,54,111,41,11,39,46,86,22,39,35,156,55,75,58,65,128,127]
    
    # draw samples from the empirical distribution
    hist, bins = np.histogram(long_diams, bins=18)
    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
    
    return random_from_cdf

def simulate_SigDparams(n):
    # draw A, b as per section 4.3 of Arias22
    
    # Case & Bhattacharya 98 
    mu, sigma = 2.04, 1.24 # mean and standard deviation
    A = np.random.normal(mu, sigma, n)
    A[A < 0] = 0.001 # no negative values 
    
    mu, sigma = 2.38, 0.26# mean and standard deviation
    b = np.random.normal(mu, sigma, n)
    
    return A, b

def Sigma_D_relation(D, A, b):
    # calculates a surface brignthess in Wm-2Hz-1sr-1, eq. 5 in Arias22
    
    sigma = A*10**(-17)*D**(-b)
    
    return sigma

def dist_to_sun(r, theta, heights):
    
    # convert simulated location into a distance to the Sun
    sun_radius = 8.5 #kpc
    # law of cosines
    d_1squared = r**2+sun_radius**2-2*r*sun_radius*np.cos(theta)
    d_proj = d_1squared**(1/2) # distance projected on the plane, save to calculate Galactic longitude
    # now pythagoras theorem with d_1 (distance in the plane) and height
    d_squared = d_1squared + heights**2
    d = d_squared**(1/2)
    # get also observed galactic latitude b
    b_rad = np.arctan(heights/(d_1squared)**(1/2)) # tan keeps sign of b
    b = 180./np.pi*b_rad
    
    return(d, b, d_proj)

def angular_size(D, d): 
    
    #small angle approximation
    theta = D/(d*1000) # D is in pc, d in kpc
    # to arcmin
    theta = theta*180/np.pi*60
    
    return theta

def location_to_gal_lat(df):
    
    r_sun = 8.5 # kpc
    # take y-projection for each point
    df['rcosalpha'] = df['r']*np.cos(df['alpha'])
    # empty column to store results
    df['gal_lat_deg'] = np.nan
    
    # 1st Gal quadrant
    # pick SNRs with y_projection (rcosalpha) < 8.5 and alpha from 0 to pi
    new_df = df.loc[(df['rcosalpha']<8.5)& (df['alpha']<np.pi)]
    cos_lat = (new_df['r']**2 - r_sun**2 - new_df['dist_proj']**2)/(-2*r_sun*new_df['dist_proj'])
    lats = np.arccos(cos_lat)*180/np.pi
    #df.loc[df[(df['rcosalpha']<8.5)& (df['alpha']<np.pi)].index,'gal_lat_deg']= lats
    df.loc[new_df.index,'gal_lat_deg']= lats

    # 4th Gal quadrant
    new_df = df.loc[(df['rcosalpha']<8.5)& (np.pi<df['alpha'])]
    cos_lat = (new_df['r']**2 - r_sun**2 - new_df['dist_proj']**2)/(-2*r_sun*new_df['dist_proj'])
    lats = -(np.arccos(cos_lat)*180/np.pi)+360
    #df.loc[df[(df['rcosalpha']<8.5)& (df['alpha']>np.pi)].index,'gal_lat_deg']= lats
    df.loc[new_df.index,'gal_lat_deg']= lats
    
    # 3rd Gal quadrant
    new_df = df.loc[(df['rcosalpha']>8.5)& (3*np.pi/2<df['alpha'])]
    cos_lat = (new_df['r']**2 - r_sun**2 - new_df['dist_proj']**2)/(-2*r_sun*new_df['dist_proj'])
    lats = -np.arccos(cos_lat)*180/np.pi
    #df.loc[df[(df['rcosalpha']>8.5)& (3*np.pi/2<df['alpha'])].index,'gal_lat_deg']= lats
    df.loc[new_df.index,'gal_lat_deg']= lats
    
    # 4th Gal quadrant
    new_df = df.loc[(df['rcosalpha']>8.5)& (df['alpha']<np.pi/2)]
    cos_lat = (new_df['r']**2 - r_sun**2 - new_df['dist_proj']**2)/(-2*r_sun*new_df['dist_proj'])
    lats = np.arccos(cos_lat)*180/np.pi
    #df.loc[df[(df['rcosalpha']>8.5)& (df['alpha']<np.pi/2)].index,'gal_lat_deg']= lats
    df.loc[new_df.index,'gal_lat_deg']= lats
    
    return df


def is_in_range(df):
    
    # sensitivity limits
    gleam_lim = 2.9*10**(-22)
    thor_lim = 10**(-22)
    green_lim = 10**(-20)
    cgps_lim = 5.3*10**(-23)
    mgps_lim = 2.9*10**(-22)
    
    #angular limits
    gleam_ang = 5
    thor_ang = 1.5
    green_ang = 3
    cgps_ang = 3
    mgps_ang = 5
    
    # galactic latitudes covered by surveys
    gleam_b_lim = [-10,10]
    thor_b_lim = [-1.25,1.25]
    green_b_lim = [-90,90]
    cgps_b_lim = [-3.6,5.6]
    mgps_b_lim = [-10,10]
    
    # angular resolution of the surveys
    gleam_ang_res = 2 # arcmin
    thor_ang_res = 1/3 # 20 arcsec
    green_ang_res = 3 # does not change
    cgps_ang_res = 1
    mgps_ang_res = 1
    
    # comparison
    sens_limits = []
    ang_limits = []
    b_limits = []
    ang_ress = []
    is_greens = []
    for lat in df['gal_lat_deg'].values:
        if 0<= lat < 17.5:
            lim = gleam_lim
            ang = gleam_ang
            bs = gleam_b_lim
            is_green = 0
            ang_res = gleam_ang_res
        elif 17.5 <= lat < 67.4:
            lim = thor_lim
            ang = thor_ang
            bs = thor_b_lim
            is_green = 0
            ang_res = thor_ang_res
        elif 67.4 <= lat < 74.2:
            lim = green_lim
            ang = green_ang
            bs = green_b_lim
            is_green = 1
            ang_res = green_ang_res
        elif 74.2 <= lat < 147.3:
            lim = cgps_lim
            ang = cgps_ang
            bs = cgps_b_lim
            ang_res = cgps_ang_res
            is_green = 0
        elif 147.3 <= lat < 245:
            lim = green_lim
            ang = green_ang
            bs = green_b_lim
            is_green = 1
            ang_res = green_ang_res
        elif 245 <= lat < 345:
            lim = mgps_lim
            ang = mgps_ang
            bs = mgps_b_lim
            is_green = 0
            ang_res = mgps_ang_res
        else:
            lim = gleam_lim
            ang = gleam_ang
            bs = green_b_lim
            is_green = 1
            ang_res = green_ang_res
        is_greens = np.append(is_greens,is_green)
        sens_limits = np.append(sens_limits,lim)
        ang_limits = np.append(ang_limits,ang)
        b_limits.append(bs)
        ang_ress = np.append(ang_ress,ang_res)
        
    df['sens_limits'] = sens_limits
    df['ang_limits'] = ang_limits
    df['b_limits'] = b_limits
    df['is_green'] = is_greens
    df['ang_res'] = ang_ress
    
    return df

def is_SNR_detected(df):
    # returns an array with 0 for non-detection and 1 for detection
    
    df = location_to_gal_lat(df)
    df = is_in_range(df)
    df = limits_per_resolution_element(df)
    
    df['detections'] = np.zeros(len(df))
    
    # if within sensitivity and angular resolution limits
    df.loc[df.loc[(df['surf_brightness']>df['sens_limits']) & (df['theta']>df['ang_limits'])].index,'detections'] = 1
    
    # but not if b outside of b_range
    for i in range(len(df)):
        if df.loc[i,'b'] > max(df.loc[i,'b_limits']):
            df.loc[i, 'detections'] = 0
        elif df.loc[i,'b'] < min(df.loc[i,'b_limits']):
            df.loc[i, 'detections'] = 0
        else:
            pass
        
    # always detected if luminosity > 10**(-20)
    nominal_sens_lim_green = 10**(-20)
    df.loc[df.loc[df['surf_brightness']>nominal_sens_lim_green].index,'detections'] = 1
    
    return df

def limits_per_resolution_element(df):
    # sensitivity to extended structure changes with the size of the source
    
    sqrtN = df['theta']/df['ang_res'] # num of resolution elements in extended source
    df['sens_limits'] = df['sens_limits']/sqrtN
    
    return df

def simulate_location(surf_brightness):
    # in polar coordinates centered at Galactic center, with a height h above the galactic plane

    n = len(surf_brightness)
    df = pd.DataFrame(columns=['surf_brightness','radii'])
    df['surf_brightness'] = surf_brightness
    
    # simulate angle
    theta = np.random.uniform(0.,2*np.pi,n)
    
    # simulate radii
    # verbene
    beta = 2.46
    r_sun = 8.5
    xi1 = np.random.uniform(0., 1, n)
    xi2 = np.random.uniform(0., 1, n)
    r = -r_sun/beta*np.log(xi1*xi2) # kpc, see discussion in Arias22
    
    # simulate height
    
    # is it CC SN or Type Ia, 3/1 odds
    where_SN_happens = np.random.binomial(1, 0.75, n) # if 1, then explosion in thin disk
    z_0 = np.zeros(len(where_SN_happens))
    z_0[where_SN_happens==1] = 0.3 # kpc, thin disk scale height
    z_0[where_SN_happens==0] = 1.5 # kpc, thick disk scale height
    
    
    R = np.random.uniform(0., 1, n)
    #z_0 = 0.3 # kpc
    heights = -z_0*np.log(1-R)
    # heights arbitrarily above/bellow the plane
    plusmin = np.random.binomial(1, 0.5, n) # one draw, 1/2 chance, n times, 1 means positive gal lat
    plusmin[plusmin == 0] = -1 # zeros means negative gal lat
    heights = heights*plusmin
    
    return r, theta, heights


def simulate_SNRs(n):
    ## randomly generates a location, Diameter, and params for Sigma-D for n SNRs in the Galaxy 
    
    diameters = simulate_D(n)
    # plt.hist(simulate_D(n), 18)
    
    A, b = simulate_SigDparams(n)
    surf_brightness = Sigma_D_relation(diameters, A, b)
    # plt.hist(np.log10(data), bins=20)
    
    r, alpha, heights = simulate_location(surf_brightness)
    
    d, b, d_proj = dist_to_sun(r, alpha, heights)
    theta = angular_size(diameters, d)
    
    # store information in a data frame
    df = pd.DataFrame(columns=['diameters','surf_brightness','r','alpha','dist','dist_proj','theta'])
    df['diameters'] = diameters
    df['surf_brightness'] = surf_brightness
    df['r'] = r
    df['alpha'] = alpha
    df['dist'] = d
    df['dist_proj'] = d_proj
    df['theta'] = theta
    df['b'] = b
    
    df = is_SNR_detected(df)
    
    return df

def find_SNRs(iterations, ns): # ns an array with the values of n
    
    # store the means and standard deviations
    means = []
    stds = []
    
    # iterate over the ns
    for n in ns:
        n = int(n)
        detections = []
        for i in range(iterations):
            df = simulate_SNRs(n)
            detections = np.append(detections,df['detections'].sum())
        means = np.append(means, detections.mean())
        stds = np.append(stds, detections.std())
            
    return means, stds

def run_simulation(iters,ns): # iters an array
        
    gap_between_ns = (int(ns[-1]-ns[0])/len(ns))/len(iters) # so not all errorbars overlap in the plot
    results = list() # list of pandas dataframes
    
    # run the simulation for each number of iterations
    for i in range(len(iters)):
        iterations = iters[i]
        df = pd.DataFrame(columns=["n","means","stds"])
        df["n"] = ns+gap_between_ns*i
        means, stds = find_SNRs(iterations, df["n"])
        df["means"] = means
        df["stds"] = stds
        results.append(df)
        
    return results

def plot_results(iters,list_of_dfs,figname=None):
    
    # for colour map
    from matplotlib.cm import get_cmap
    name = "Dark2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors 

    # plot the results
    fig = plt.figure(figsize=(7,4))
    for i, df in zip(range(len(iters)),list_of_dfs):
        plt.plot(df["n"], df["means"]/df["n"], color=colors[i], label='{} iterations'.format(iters[i]))
        plt.errorbar(df["n"], df["means"]/df["n"], yerr=df["stds"]/df["n"], color=colors[i], alpha=0.75)
    plt.xlabel("Count of simulated SNRs ($n$)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Fraction of detected SNRs", fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Results of simulations", fontsize=18)
    plt.legend(loc='lower right', fontsize=12)
    if figname==None:
        fig.savefig('./Figs/simulation_results.pdf', dpi=100, bbox_inches="tight", pad_inches=1)
    else:
        fig.savefig('./Figs/'+figname+'.pdf', dpi=100, bbox_inches="tight", pad_inches=1)
        
# what is the confusion limit -- how many SNRs are overlapping and in the same line of sight
# load list of HII regions
hii = pd.read_csv('hii_selected.csv', index_col=0)


def whatstheconfusion(iters,n):
    
    #confused_noscatter = []
    confused = []
    confused_with_hii = []
    
    for iteration in range(iters):
        
        df2 = simulate_SNRs(n)
        df1 = df2.copy() # make a copy for HII region study
        df = df2
        
        # first, confusion just between SNRs
        all_dis = cdist(df[['b','gal_lat_deg']], df[['b','gal_lat_deg']], 'euclidean') # distances between every point
        nneighbour_theta = []
        nneighbour_dist = []
        for i in range(len(df)):
            dists = all_dis[i] # distances from point i in the df to all other points (incl. itself)
            s = set(dists)
            min_dist = sorted(s)[1] # because actual min dist is 0, point to itself
            neighbour_index = np.where(dists == min_dist)
            nn_size = df.iloc[neighbour_index[0]]['theta'].values[0]
            nneighbour_dist = np.append(nneighbour_dist,min_dist*60 )# dist to arcmin
            nneighbour_theta = np.append(nneighbour_theta,nn_size)
        df['nneighbour_dist'] = nneighbour_dist
        df['nneighbour_theta'] = nneighbour_theta
        # check if source is confused
        df['is_confused'] = np.zeros(len(df))
        # 1 when r_1 + r_2 < d (overlapping)
        df.loc[df.loc[df['nneighbour_dist']<(df['nneighbour_theta']/2+df['theta']/2)].index,'is_confused'] = 1
        # confused if is_confused and detected
        number_of_confused = len(df.loc[(df['is_confused']==1)&(df['detections']==1)])
        confused = np.append(confused,number_of_confused)
        
        # now what if we add H II regions
        df = df1
        df['is_snr'] = 1 # save SNRs apart from HII regions
        df = pd.concat([df, hii], axis=0)
        all_dis = cdist(df[['b','gal_lat_deg']], df[['b','gal_lat_deg']], 'euclidean') # distances between every point
        nneighbour_theta = []
        nneighbour_dist = []
        for i in range(len(df)):
            if df.iloc[i]["is_snr"] != 1: # no need to see if HII regions are confused
                pass
            else:
                dists = all_dis[i] # distances from point i in the df to all other points (incl. itself)
                s = set(dists)
                min_dist = sorted(s)[1] # because actual min dist is 0, point to itself
                neighbour_index = np.where(dists == min_dist)
                nn_size = df.iloc[neighbour_index[0]]['theta'].values[0]
                nneighbour_dist = np.append(nneighbour_dist,min_dist*60 )# dist to arcmin
                nneighbour_theta = np.append(nneighbour_theta,nn_size)
        df = df.loc[df['is_snr']==1] # remove HII regs from df
        df['nneighbour_dist'] = nneighbour_dist
        df['nneighbour_theta'] = nneighbour_theta
        # check if source is confused
        df['is_confused'] = np.zeros(len(df))
        df.loc[df.loc[df['nneighbour_dist']<(df['nneighbour_theta']/2+df['theta']/2)].index,'is_confused'] = 1
        # confused if is_confused and detected AND a snr
        number_of_confused = len(df.loc[(df['is_confused']==1)&(df['detections']==1)&(df['is_snr']==1)])
        confused_with_hii = np.append(confused_with_hii,number_of_confused)
    
    return confused.mean()/n,confused.std(), confused_with_hii.mean()/n,confused_with_hii.std()


def confusion_plots(iterations,ns):

    conf_fraction_solo = []
    conf_fraction_withhii = []
    std_solo = []
    std_withhii = []
    
    for n in ns:
        frac_solo, solo_sdt, frac_hii, hii_sdt = whatstheconfusion(iterations,n)
        conf_fraction_solo = np.append(conf_fraction_solo,frac_solo)
        std_solo = np.append(std_solo,solo_sdt)
        conf_fraction_withhii = np.append(conf_fraction_withhii,frac_hii)
        std_withhii = np.append(std_withhii,hii_sdt)
    
    # plot
    # for colour map
    from matplotlib.cm import get_cmap
    name = "Dark2"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors 

    plt.figure(figsize=(7,4))
    plt.plot(ns,conf_fraction_solo,label="Confusion SNRs (no HII)", color=colors[0])
    plt.errorbar(ns,conf_fraction_solo,yerr=solo_sdt/ns, color=colors[0])
    plt.plot(ns,conf_fraction_withhii, label='Confusion incl. HII regs.', color=colors[3])
    plt.errorbar(ns,conf_fraction_withhii,yerr=std_withhii/ns, color=colors[3])
    plt.xlabel("Count of simulated SNRs ($n$)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Fraction of confused SNRs", fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.title("Effect of confusion", fontsize=18)
    plt.savefig('./Figs/confusion.pdf', dpi=100, bbox_inches="tight", pad_inches=1)