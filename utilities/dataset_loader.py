import numpy as np
import pandas as pd
import sys
from tqdm import tqdm, trange
import time
import scipy
import os, shutil
sys.path.append("Data")
# import airfoil_utils

def validate_and_correct(X, N, P, f, correct):
    if correct:
        suffix = " Correcting..."
    else:
        suffix = ""
    if X is not None:
        rx = f(X)
        X_error = len(rx) - np.sum(rx).astype(int)
        if X_error>0:
            print(f"Warning: {X_error} X values misclassified by validity function.{suffix}")
        if correct:
            X = X[rx]
    if N is not None:
        rn = f(N)
        N_error = np.sum(rn).astype(int)
        if N_error>0:
            print(f"Warning: {N_error} N values misclassified by validity function.{suffix}")
        if correct:
            N = N[~rn]
    if not P is None:
        rp = f(P)
        P_error = len(rp) - np.sum(rp).astype(int)
        if P_error>0:
            print(f"Warning: {P_error} P values misclassified by validity function.{suffix}")
        if correct:
            P = P[rp]
    return X, N, P


def dataset_properties(X, N, P, f, num_eval = 1000000):
    lr = np.quantile(X, 0.00, axis=0)
    ur = np.quantile(X, 1.0, axis=0)
    randoms = np.random.rand(num_eval, X.shape[1])
    randoms = randoms*(ur-lr) + lr
    start_time = time.time()
    
    validity = f(randoms)
    end_time = time.time()
    eval_time = (end_time - start_time)/num_eval
    validity_count = np.sum(validity)
    validity_rate = validity_count/num_eval
    num_dist = X.shape[0]
    num_neg = N.shape[0]
    num_pos = P.shape[0]
    return eval_time, validity_rate, num_dist, num_neg, num_pos

def load_framed_validity(datadir):
    sys.path.append(datadir)
    import framed_validity
    clf_data = pd.read_csv(f"{datadir}Framed.csv", index_col=0)
    clf_data = clf_data.drop(["Material"], axis=1)
    clf_data = clf_data.drop(["valid"], axis=1)
    columns = clf_data.columns
    def validity_fn(_x, columns = columns):
        _x_df = pd.DataFrame(_x, columns=columns)
        idx_s, prec =  framed_validity.getInvalid(_x_df)
        validity = np.ones(len(_x))
        validity[idx_s] = 0
        return validity.astype(bool)
    return validity_fn

def get_load_framed(datadir):
    def load_framed(datadir=datadir):
        clf_data = pd.read_csv(f"{datadir}Framed.csv", index_col=0)
        clf_data = clf_data.drop(["Material"], axis=1)
        PGN_mask = np.char.startswith(np.array(clf_data.index).astype(str), "Gen")
        N_mask = clf_data["valid"] == 1
        P_mask = np.logical_and(PGN_mask, ~N_mask)
        X_mask = np.logical_and(~PGN_mask, ~N_mask)
        clf_data = clf_data.drop(["valid"], axis=1)
        N = clf_data[N_mask]
        P = clf_data[P_mask]
        X = clf_data[X_mask]
        validity_fn = load_framed_validity(datadir)
        X = np.array(X)
        N = np.array(N)
        P = np.array(P)
        X, N, P = validate_and_correct(X, N, P, validity_fn, True)
        return X, N, P
    return load_framed


def load_shipd_validity(datadir):
    sys.path.append(datadir)
    import shipd_validity
    def validity_fn(_x):
        #append a column of 10s to the front of the array
        _x = np.concatenate((np.ones((_x.shape[0], 1))*10, _x), axis=1)
        all_cons = []
        for i in trange(0, len(_x)):
            hull = shipd_validity.Hull_Parameterization(_x[i,:])
            cons = hull.input_Constraints()
            all_cons.append(np.any(np.greater(cons, 0)))
        all_cons = np.array(all_cons).astype(bool)
        return ~all_cons
    return validity_fn
def get_load_shipd(datadir):
    def load_shipd(datadir=datadir):
        d1 = pd.read_csv(f"{datadir}Shipd_1.csv")
        d2 = pd.read_csv(f"{datadir}Shipd_2.csv")
        d3 = pd.read_csv(f"{datadir}Shipd_3.csv")
        X = pd.concat([d1, d2, d3], axis=0)
        X = X.drop(X.columns[0], axis=1)
        N = np.load(f"{datadir}Shipd_n.npy")
        N = pd.DataFrame(N, columns=X.columns)
        P = None
        validity_fn = load_shipd_validity(datadir)
        X = np.array(X)
        N = np.array(N)
        X, N, P = validate_and_correct(X, N, P, validity_fn, True)
        return X, N, P
    return load_shipd






#---------------------------------------------------------------------------------------------------------





















import torch
import numpy as np
from botorch.test_functions import Ackley
from scipy.stats import qmc

# Individuals should be in the range of -10, 10

def Ackley2D(individuals): # This should be the test function
    
    #############################################################################
    #############################################################################
    # Set function here:
    dimm = 2
    dtype = torch.double
    device = torch.device("cpu")
    fun = Ackley(dim=dimm, negate=True).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds
    #############################################################################
    #############################################################################
    
    
    n = individuals.size(0)

    fx = fun(individuals)
    fx = fx.reshape((n, 1))

    #############################################################################
    ## Constraints
    gx1 = torch.sum(individuals,1)  # sigma(x) <= 0 
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.norm(individuals, p=2, dim=1)-5  # norm_2(x) -3 <= 0
    gx2 = gx2.reshape((n, 1))

    gx = torch.cat((gx1, gx2), 1)
    #############################################################################
    
    
    return gx, fx

def Ackley2D_Scaling(X):
    
    X_scaled = X*15-5
    
    return X_scaled

def Ackley6D(individuals): # This should be the test function
    
    #############################################################################
    #############################################################################
    # Set function here:
    dimm = 6
    dtype = torch.double
    device = torch.device("cpu")
    fun = Ackley(dim=dimm, negate=True).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds
    #############################################################################
    #############################################################################
    
    
    n = individuals.size(0)

    fx = fun(individuals)
    fx = fx.reshape((n, 1))

    #############################################################################
    ## Constraints
    gx1 = torch.sum(individuals,1)  # sigma(x) <= 0 
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.norm(individuals, p=2, dim=1)-5  # norm_2(x) -3 <= 0
    gx2 = gx2.reshape((n, 1))

    gx = torch.cat((gx1, gx2), 1)
    #############################################################################
    
    
    return gx, fx

def Ackley6D_Scaling(X):
    
    X_scaled = X*15-5
    
    return X_scaled

def Ackley10D(individuals): # This should be the test function
    
    #############################################################################
    #############################################################################
    # Set function here:
    dimm = 10
    dtype = torch.double
    device = torch.device("cpu")
    fun = Ackley(dim=dimm, negate=True).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds
    #############################################################################
    #############################################################################
    
    
    n = individuals.size(0)

    fx = fun(individuals)
    fx = fx.reshape((n, 1))

    #############################################################################
    ## Constraints
    gx1 = torch.sum(individuals,1)  # sigma(x) <= 0 
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.norm(individuals, p=2, dim=1)-5  # norm_2(x) -3 <= 0
    gx2 = gx2.reshape((n, 1))

    gx = torch.cat((gx1, gx2), 1)
    #############################################################################
    
    
    return gx, fx

def Ackley10D_Scaling(X):
    
    X_scaled = X*15-5
    
    return X_scaled
def Ashbychart_wrapper(datadir): # This should be the test function
    def Ashbychart(individuals, datadir=datadir): # This should be the test function

        


        Pcr = 10
        L = 2
        C = 1
        
        rho = individuals[:,0]
        Sy   = individuals[:,1]
        E   = individuals[:,2]
        fx = (- rho*L*2*Pcr/Sy )
        gA = (3*L**2 * Sy**2) / (C*np.pi*np.pi*Pcr) - E
        gA = np.expand_dims(gA, axis=1)
        gA = torch.tensor(gA)

        individuals_inverse_scale = Ashbychart_Inverse_Scaling(individuals)
        pixels_x = Ashbychart_load_pixelX_for_gc(individuals_inverse_scale)
        gc = -Ashbychart_get_gc(pixels_x, datadir)

        gA = torch.cat((gA, gc), 1)
        fx = torch.tensor(fx)
        
        # fx = torch.tensor(fx)
        # fx = torch.reshape(fx, (len(fx),1))
        # gA = torch.tensor(gA)
        # gA = torch.reshape(gA, (len(fx),1))

        return gA, fx
    return Ashbychart

def Ashbychart_Scaling(X): # This should be the test function

    l_bounds = [10, 0.01, 1e-4 ]
    u_bounds = [50000,10000,1000]
    X_scaled = qmc.scale(X, l_bounds, u_bounds)

    return X_scaled

def Ashbychart_Inverse_Scaling(X): # This should be the test function
    l_bounds = torch.tensor([10.0, 0.01, 1e-4 ])
    u_bounds = torch.tensor([50000.0,10000.0,1000.0])
    divisor = torch.subtract(u_bounds, l_bounds)
    X_scaled = torch.divide(X-l_bounds, divisor)
    return X_scaled

def Ashbychart_load_pixelX_for_gc(X):
    
    # print(X)
    
    rho = (X[:,0]  * (np.log10(50000)-np.log10(10  )) + np.log10(10    )   ).reshape(X.shape[0],1)
    Sy  = (X[:,1]  * (np.log10(10000)-np.log10(0.01  )) + np.log10(0.01  )   ).reshape(X.shape[0],1)
    E   = (X[:,2]  * (np.log10(1000 )-np.log10(1e-4)) + np.log10(1e-4  )   ).reshape(X.shape[0],1)
    
    rho = torch.tensor(rho).reshape(X.shape[0],1)
    Sy = torch.tensor(Sy).reshape(X.shape[0],1)
    E = torch.tensor(E).reshape(X.shape[0],1)
    

    X_unnormed = torch.cat((rho, Sy, E), dim=1)
    
    # First Ashby Chart
    left_bound = 215
    right_bound = 795
    bottom_bound = 95
    up_bound = 472

    rhoE_x_scale = (np.log10(50000)-np.log10(10)) / (right_bound-left_bound)
    rhoE_y_scale = (np.log10(1000)-np.log10(1e-4)) / (up_bound-bottom_bound)

    log_rho = torch.round( ( rho - np.log10(10) )/rhoE_x_scale + left_bound )
    log_E = torch.round( up_bound- ( E -np.log10(1e-4) )/rhoE_y_scale )

    # Second Ashby Chart
    left_bound = 215
    right_bound = 795
    bottom_bound = 55
    up_bound = 472
    rhoSy_y_scale = (np.log10(10000)-np.log10(1e-2)) / (up_bound-bottom_bound)

    log_Sy =  torch.round( up_bound- ( Sy - np.log10(1e-2) )/rhoSy_y_scale )

    pixel_X = torch.cat((log_rho, log_Sy, log_E), dim=1)
    
    return pixel_X

import matplotlib.image as mpimg

def Ashbychart_get_gc(pixel_X, datadir):
 
    # Read Images
    SyvsRho_gray_img = mpimg.imread(f'{datadir}SyvsRho_gray.png')
    EvsRho_gray_img = mpimg.imread(f'{datadir}EvsRho_gray.png')
    
    gc = torch.zeros(pixel_X.shape[0], 1)
    ind = 0
    
    for x in pixel_X:
        rho = int(x[0])
        Sy  = int(x[1])
        E   = int(x[2])
        if (SyvsRho_gray_img[Sy,rho,1] != 1)  & (EvsRho_gray_img[E,rho,1] != 1) :
            gc[ind]=1
        else:
            gc[ind]=-1
        ind+=1
    
    
    return gc

def CantileverBeam(individuals): # This should be the test function
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    gx5 = []
    gx6 = []
    gx7 = []
    gx8 = []
    gx9 = []
    gx10 = []
    gx11 = []
    
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        
        P = 50000
        E = 2*107
        L = 100
        
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( x1*x6*L + x2*x7*L + x3*x8*L + x4*x9*L + x5*x10*L )
        # test_function = - ( C1*C2 - C3 + C4 + C5 )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 600 * P / (x5*x10*x10) - 14000
        g2 = 6 * P * (L*2) / (x4*x9*x9) - 14000
        g3 = 6 * P * (L*3) / (x3*x8*x8) - 14000
        g4 = 6 * P * (L*4) / (x2*x7*x7) - 14000
        g5 = 6 * P * (L*5) / (x1*x6*x6) - 14000
        g6 = P* L**3 * (1/L + 7/L + 19/L + 37/L + 61/L) / (3*E) -2.7
        g7 = x10/x5 - 20
        g8 = x9/x4 - 20
        g9 = x8/x3 - 20
        g10 = x7/x2 - 20
        g11 = x6/x1 - 20

        
        
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
        gx5.append( g5 )       
        gx6.append( g6 )    
        gx7.append( g7 )            
        gx8.append( g8 )
        gx9.append( g9 )
        gx10.append( g10 )
        gx11.append( g11 )
    
    # print(gx5)
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    gx4 = torch.tensor(gx4)  
    gx4 = gx4.reshape((n, 1))
    
    gx5 = torch.tensor(gx5)  
    gx5 = gx1.reshape((n, 1))

    gx6 = torch.tensor(gx6)  
    gx6 = gx2.reshape((n, 1))
    
    gx7 = torch.tensor(gx7)  
    gx7 = gx3.reshape((n, 1))
    
    gx8 = torch.tensor(gx8)  
    gx8 = gx4.reshape((n, 1))
    
    gx9 = torch.tensor(gx9)  
    gx9 = gx4.reshape((n, 1))
    
    gx10 = torch.tensor(gx10)  
    gx10 = gx4.reshape((n, 1))
    
    gx11 = torch.tensor(gx11)  
    gx11 = gx4.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11), 1)
    #############################################################################
    #############################################################################
    
    
    return gx, fx

def CantileverBeam_Scaling(X):
    x1 = (X[:,0] * (5-1) + 1).reshape(X.shape[0],1)
    x2 = (X[:,1] * (5-1) + 1).reshape(X.shape[0],1)
    x3 = (X[:,2] * (5-1) + 1).reshape(X.shape[0],1)
    x4 = (X[:,3] * (5-1) + 1).reshape(X.shape[0],1)
    x5 = (X[:,4] * (5-1) + 1).reshape(X.shape[0],1)
    x6 = (X[:,5] * (65-30) + 30).reshape(X.shape[0],1)
    x7 = (X[:,6] * (65-30) + 30).reshape(X.shape[0],1)
    x8 = (X[:,7] * (65-30) + 30).reshape(X.shape[0],1)
    x9 = (X[:,8] * (65-30) + 30).reshape(X.shape[0],1)
    x10 = (X[:,9] * (65-30) + 30).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), dim=1)
    return X_scaled

def Car(individuals): # This should be the test function
    
    n = individuals.size(0)
    
    fx = torch.zeros((n,1))
    gx1 = torch.zeros((n,1))
    gx2 = torch.zeros((n,1))
    gx3 = torch.zeros((n,1))
    gx4 = torch.zeros((n,1))
    gx5 = torch.zeros((n,1))
    gx6 = torch.zeros((n,1))
    gx7 = torch.zeros((n,1))
    gx8 = torch.zeros((n,1))
    gx9 = torch.zeros((n,1))
    gx10 = torch.zeros((n,1))
    gx11 = torch.zeros((n,1))
    
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]

        
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( 1.98 + 4.90*x1 + 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 2.73*x7 )
        
        ## Calculate constraints terms 
        g1 = 1.16 - 0.3717*x2*x4 - 0.00931*x2*x10 - 0.484*x3*x9 + 0.01343*x6*x10   -1
        
        g2 = (0.261 - 0.0159*x1*x2 - 0.188*x1*x8 
              - 0.019*x2*x7 + 0.0144*x3*x5 + 0.0008757*x5*x10
              + 0.08045*x6*x9 + 0.00139*x8*x11 + 0.00001575*x10*x11)        -0.9
        
        g3 = (0.214 + 0.00817*x5 - 0.131*x1*x8 - 0.0704*x1*x9 + 0.03099*x2*x6
              -0.018*x2*x7 + 0.0208*x3*x8 + 0.121*x3*x9 - 0.00364*x5*x6
              +0.0007715*x5*x10 - 0.0005354*x6*x10 + 0.00121*x8*x11)        -0.9
        
        g4 = 0.74 -0.061*x2 -0.163*x3*x8 +0.001232*x3*x10 -0.166*x7*x9 +0.227*x2*x2        -0.9
        
        g5 = 28.98 +3.818*x3-4.2*x1*x2+0.0207*x5*x10+6.63*x6*x9-7.7*x7*x8+0.32*x9*x10    -32
        
        g6 = 33.86 +2.95*x3+0.1792*x10-5.057*x1*x2-11.0*x2*x8-0.0215*x5*x10-9.98*x7*x8+22.0*x8*x9    -32
        
        g7 = 46.36 -9.9*x2-12.9*x1*x8+0.1107*x3*x10    -32
        
        g8 = 4.72 -0.5*x4-0.19*x2*x3-0.0122*x4*x10+0.009325*x6*x10+0.000191*x11**2     -4
        
        g9 = 10.58 -0.674*x1*x2-1.95*x2*x8+0.02054*x3*x10-0.0198*x4*x10+0.028*x6*x10     -9.9
        
        g10 = 16.45 -0.489*x3*x7-0.843*x5*x6+0.0432*x9*x10-0.0556*x9*x11-0.000786*x11**2     -15.7
        
        
        gx1[i] = g1
        gx2[i] = g2
        gx3[i] = g3
        gx4[i] = g4
        gx5[i] = g5
        gx6[i] = g6
        gx7[i] = g7
        gx8[i] = g8
        gx9[i] = g9
        gx10[i] = g10
        fx[i] = test_function
        
        
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10), 1)
    #############################################################################
    #############################################################################

    
    return gx, fx

def Car_Scaling(X):
    x1  = (X[:,0] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x2  = (X[:,1] * (1.35-0.45) + 0.45).reshape(X.shape[0],1)
    x3  = (X[:,2] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x4 = (X[:,3] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x5 = (X[:,4] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x6 = (X[:,5] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x7 = (X[:,6] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x8 = (X[:,7] * (0.345-0.192) + 0.192).reshape(X.shape[0],1)
    x9 = (X[:,8] * (0.345-0.192) + 0.192).reshape(X.shape[0],1)
    x10 = (X[:,9] * (-20)).reshape(X.shape[0],1)
    x11 = (X[:,10] * (-20)).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), dim=1)
    return X_scaled

def CompressionSpring(individuals): # This should be the test function
    
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        d = x[0]
        D = x[1]
        N = x[2]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( (N+2)*D*d**2 )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 1 -  ( D*D*D * N / (71785* d*d*d*d) )
        g2 = (4*D*D - D*d) / (12566 * (D*d*d*d - d*d*d*d)) + 1/(5108*d*d) -  1
        g3 = 1 - 140.45*d / (D*D * N)
        g4 = (D+d)/1.5 - 1
        
        
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
       
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    gx4 = torch.tensor(gx4)  
    gx4 = gx4.reshape((n, 1))
    
    gx = torch.cat((gx1, gx2, gx3, gx4), 1)
    #############################################################################
    #############################################################################

    
    return gx, fx

def CompressionSpring_Scaling(X): 
    
    d = (X[:,0] * ( 1   - 0.05 ) + 0.05 ).reshape(X.shape[0],1)
    D = (X[:,1] * ( 1.3 - 0.25 ) + 0.25   ).reshape(X.shape[0],1)
    N = (X[:,2]  * ( 15  - 2    ) + 2         ).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((d, D, N), dim=1)

    return X_scaled

def HeatExchanger(individuals): # This should be the test function
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    gx5 = []
    gx6 = []
    gx7 = []
    gx8 = []
    gx9 = []
    gx10 = []
    gx11 = []
    
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( x1+x2+x3 )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 0.0025 * (x4+x6) - 1
        g2 = 0.0025 * (x5 + x7 - x4) - 1
        g3 = 0.01 *(x8-x5) - 1
        g4 = 833.33252*x4 + 100*x1 - x1*x6 - 83333.333
        g5 = 1250*x5 + x2*x4 - x2*x7 - 125*x4
        g6 = x3*x5 - 2500*x5 - x3*x8 + 125*10000


        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
        gx5.append( g5 )       
        gx6.append( g6 )    
    
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    gx4 = torch.tensor(gx4)  
    gx4 = gx4.reshape((n, 1))
    
    gx5 = torch.tensor(gx5)  
    gx5 = gx1.reshape((n, 1))

    gx6 = torch.tensor(gx6)  
    gx6 = gx2.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6), 1)

    #############################################################################
    #############################################################################

    return gx, fx

def HeatExchanger_Scaling(X):
    
    x1 = (X[:,0] * (10000-100) + 100).reshape(X.shape[0],1)
    x2 = (X[:,1] * (10000-1000) + 1000).reshape(X.shape[0],1)
    x3 = (X[:,2] * (10000-1000) + 1000).reshape(X.shape[0],1)
    x4 = (X[:,3] * (1000-10) + 10).reshape(X.shape[0],1)
    x5 = (X[:,4] * (1000-10) + 10).reshape(X.shape[0],1)
    x6 = (X[:,5] * (1000-10) + 10).reshape(X.shape[0],1)
    x7 = (X[:,6] * (1000-10) + 10).reshape(X.shape[0],1)
    x8 = (X[:,7] * (1000-10) + 10).reshape(X.shape[0],1)
    
    
    X_scaled = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
    
    return X_scaled

def PressureVessel(individuals): # This should be the test function
    
    C1 = 0.6224
    C2 = 1.7781
    C3 = 3.1661
    C4 = 19.84
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        
        Ts = x[0]
        Th = x[1]
        R = x[2]
        L = x[3]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( C1*Ts*R*L + C2*Th*R*R + C3*Ts*Ts*L + C4*Ts*Ts*R )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = -Ts + 0.0193*R
        g2 = -Th + 0.00954*R
        g3 = (-1)*np.pi*R*R*L + (-1)*4/3*np.pi*R*R*R + 750*1728
        
        g4 = L-240
        
        
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
       
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    gx4 = torch.tensor(gx4)  
    gx4 = gx4.reshape((n, 1))
    
    gx = torch.cat((gx1, gx2, gx3, gx4), 1)
    #############################################################################
    #############################################################################
    
    return gx, fx

def PressureVessel_Scaling(X):
    
    Ts  = (X[:,0] * (98*0.0625) + 0.0625).reshape(X.shape[0],1)
    Th  = (X[:,1] * (98*0.0625) + 0.0625).reshape(X.shape[0],1)
    R   = (X[:,2] * (200-10) + 10).reshape(X.shape[0],1)
    L   = (X[:,3] * (200-10) ).reshape(X.shape[0],1)
    
    
    X_scaled = torch.cat((Ts, Th, R, L), dim=1)
    
    return X_scaled

def ReinforcedConcreteBeam(individuals): # This should be the test function
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        As = x[0]
        h = x[1]
        b = x[2]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( 29.4*As + 0.6*b*h )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = h/b - 4
        g2 = 180 + 7.35*As*As/b - As*h
        
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    

       
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2), 1)
    #############################################################################
    #############################################################################
    
    
    return gx, fx
    
def ReinforcedConcreteBeam_Scaling(X):
    As = (X[:,0] * (15-0.2) + 0.2).reshape(X.shape[0],1)
    b  = (X[:,1] * (40-28)  +28).reshape(X.shape[0],1)
    h  = (X[:,2] * 5 + 5).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((As, b, h), dim=1)
    return X_scaled    

def SpeedReducer(individuals): # This should be the test function
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    gx5 = []
    gx6 = []
    gx7 = []
    gx8 = []
    gx9 = []
    gx10 = []
    gx11 = []
    
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        b = x[0]
        m = x[1]
        z = x[2]
        L1 = x[3]
        L2 = x[4]
        d1 = x[5]
        d2 = x[6]
        
        C1 = 0.7854*b*m*m
        C2 = 3.3333*z*z + 14.9334*z - 43.0934
        C3 = 1.508*b*(d1*d1 + d2*d2)
        C4 = 7.4777*(d1*d1*d1 + d2*d2*d2)
        C5 = 0.7854*(L1*d1*d1 + L2*d2*d2)
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( 0.7854*b*m*m * (3.3333*z*z + 14.9334*z - 43.0934) - 1.508*b*(d1*d1 + d2*d2) + 7.4777*(d1*d1*d1 + d2*d2*d2) + 0.7854*(L1*d1*d1 + L2*d2*d2)  )

        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 27/(b*m*m*z) - 1
        g2 = 397.5/(b*m*m*z*z) - 1
        
        
        g3 = 1.93*L1**3 /(m*z *d1**4) - 1
        g4 = 1.93*L2**3 /(m*z *d2**4) - 1
        
        
        
        g5 = np.sqrt( (745*L1/(m*z))**2 + 1.69*1e6 ) / (110*d1**3) -1
        g6 = np.sqrt( (745*L2/(m*z))**2 + 157.5*1e6 ) / (85*d2**3) -1
        g7 = m*z/40 - 1
        g8 = 5*m/(b) - 1
        g9 = b/(12*m) -1

        
        
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
        gx5.append( g5 )       
        gx6.append( g6 )    
        gx7.append( g7 )            
        gx8.append( g8 )
        gx9.append( g9 )

    
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    gx4 = torch.tensor(gx4)  
    gx4 = gx4.reshape((n, 1))
    
    gx5 = torch.tensor(gx5)  
    gx5 = gx1.reshape((n, 1))

    gx6 = torch.tensor(gx6)  
    gx6 = gx2.reshape((n, 1))
    
    gx7 = torch.tensor(gx7)  
    gx7 = gx3.reshape((n, 1))
    
    gx8 = torch.tensor(gx8)  
    gx8 = gx4.reshape((n, 1))
    
    gx9 = torch.tensor(gx9)  
    gx9 = gx4.reshape((n, 1))

    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9), 1)

    #############################################################################
    #############################################################################
    
    
    return gx, fx

def SpeedReducer_Scaling(X):

    b  = (X[:,0] * ( 3.6 - 2.6 ) + 2.6).reshape(X.shape[0],1)
    m  = (X[:,1] * ( 0.8 - 0.7 ) + 0.7).reshape(X.shape[0],1)
    z  = (X[:,2] * ( 28 - 17 ) + 17).reshape(X.shape[0],1)
    L1 = (X[:,3] * ( 8.3 - 7.3 ) + 7.3).reshape(X.shape[0],1)
    L2 = (X[:,4] * ( 8.3 - 7.3 ) + 7.3).reshape(X.shape[0],1)
    d1 = (X[:,5] * ( 3.9 - 2.9 ) + 2.9).reshape(X.shape[0],1)
    d2 = (X[:,6] * ( 5.5 - 5 ) + 5).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((b, m, z, L1, L2, d1, d2), dim=1)
    return X_scaled

def ThreeTruss(individuals): # This should be the test function
    

    fx = []
    gx1 = []
    gx2 = []
    gx3 = []

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        x1 = x[0]
        x2 = x[1]
        L = 100
        P = 2
        sigma = 2
        
        ## Negative sign to make it a maximization problem
        test_function = - ( 2*np.sqrt(2)*x1 + x2 ) * L
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = ( np.sqrt(2)*x1 + x2 ) / (np.sqrt(2)*x1*x1 + 2*x1*x2) * P - sigma
        g2 = ( x2 ) / (np.sqrt(2)*x1*x1 + 2*x1*x2) * P - sigma
        g3 = ( 1 ) / (x1 + np.sqrt(2)*x2) * P - sigma
    
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
       
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2, gx3), 1)
    #############################################################################
    #############################################################################

    return gx, fx

def ThreeTruss_Scaling(X):

    return X

def ToyCole1(individuals): # This should be the test function
    fx = []
    gx = []
    
    for x in individuals:
        test_function = (- (x[0]-0.5)**2 - (x[1]-0.5)**2 ) 
        fx.append(test_function) 
        gx.append( x[0] + x[1] - 0.75 )
        
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))
    gx = torch.tensor(gx)
    gx = torch.reshape(gx, (len(gx),1))
    
    return gx, fx

def ToyCole1_Scaling(X): # This should be the test function

    
    return X

def ToyCole2(individuals): 
    fx = []
    gx = []
    
    for x in individuals:
        
        ## Negative sign to make it a maximization problem
        test_function = - ( np.cos(2*x[0])*np.cos(x[1]) +  np.sin(x[0]) ) 
        fx.append(test_function) 
        gx.append( ((x[0]+5)**2)/4 + (x[1]**2)/100 -2.5 )
        
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))
    gx = torch.tensor(gx)
    gx = torch.reshape(gx, (len(gx),1))
    
    return gx, fx

def ToyCole2_Scaling(individuals): 
    X = individuals
    X1 = X[:,0].reshape(X.size(0),1)
    X1 = X1*5-5
    X2 = X[:,1].reshape(X.size(0),1)
    X2 = X2*10-5
    X_scaled = torch.tensor(np.concatenate((X1,X2), axis=1))
    
    return X_scaled

def ToyGardner1(individuals): # This should be the test function
    fx = []
    gx = []
    for x in individuals:
        g = np.cos(x[0])*np.cos(x[1]) -  np.sin(x[0])*np.sin(x[1]) -0.5
        fx.append( - np.cos(2*x[0])*np.cos(x[1]) -  np.sin(x[0])  ) 
        gx.append( g )
        
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))
    gx = torch.tensor(gx)
    gx = torch.reshape(gx, (len(gx),1))
    return gx, fx

def ToyGardner1_Scaling(X):
    X_scaled = X*6;
    return X_scaled

def ToyGardner2(individuals): 
    fx = []
    gx = []
    
    for x in individuals:
        
        g = np.sin(x[0])*np.sin(x[1]) + 0.95
        fx.append( - np.sin(x[0]) - x[1]  ) # maximize -(x1^2 +x 2^2)
        gx.append( g )
        
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))
    gx = torch.tensor(gx)
    gx = torch.reshape(gx, (len(gx),1))
    
    return gx, fx

def ToyGardner2_Scaling(X):
    X_scaled = X*6;
    return X_scaled

def WeldedBeam(individuals): # This should be the test function
    
    
    C1 = 1.10471
    C2 = 0.04811
    C3 = 14.0
    fx = torch.zeros(individuals.shape[0], 1)
    gx1 = torch.zeros(individuals.shape[0], 1)
    gx2 = torch.zeros(individuals.shape[0], 1)
    gx3 = torch.zeros(individuals.shape[0], 1)
    gx4 = torch.zeros(individuals.shape[0], 1)
    gx5 = torch.zeros(individuals.shape[0], 1)
    
    for i in range(individuals.shape[0]):
        
        x = individuals[i,:]

        h = x[0]
        l = x[1]
        t = x[2]
        b = x[3]
        
        test_function = - ( C1*h*h*l + C2*t*b*(C3+l) )
        fx[i] = test_function
        
        ## Calculate constraints terms 
        tao_dx = 6000 / (np.sqrt(2)*h*l)
        
        tao_dxx = 6000*(14+0.5*l)*np.sqrt( 0.25*(l**2 + (h+t)**2 ) ) / (2* (0.707*h*l * ( l**2 /12 + 0.25*(h+t)**2 ) ) )
        
        tao = np.sqrt( tao_dx**2 + tao_dxx**2 + l*tao_dx*tao_dxx / np.sqrt(0.25*(l**2 + (h+t)**2)) )
        
        sigma = 504000/ (t**2 * b)
        
        P_c = 64746*(1-0.0282346*t)* t * b**3
        
        delta = 2.1952/ (t**3 *b)
        
        
        ## Calculate 5 constraints
        g1 = (-1) * (13600- tao) 
        g2 = (-1) * (30000 - sigma) 
        g3 = (-1) * (b - h)
        g4 = (-1) * (P_c - 6000) 
        g5 = (-1) * (0.25 - delta)
        
        gx1[i] =  g1        
        gx2[i] =  g2     
        gx3[i] =  g3             
        gx4[i] =  g4 
        gx5[i] =  g5 
    
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5), 1)
    return gx, fx

def WeldedBeam_Scaling(X):
    h = (X[:,0]  * (10-0.125) + 0.125 ).reshape(X.shape[0],1)
    l = (X[:,1]  * (15-0.1  ) + 0.1   ).reshape(X.shape[0],1)
    t = (X[:,2]  * (10-0.1 ) + 0.1         ).reshape(X.shape[0],1)
    b = (X[:,3]  * (10-0.1 ) + 0.1         ).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((h, l, t, b), dim=1)
    return X_scaled

def lookup_function(function_ID, datadir):
    function_dict = {"Ackley2D":[Ackley2D, Ackley2D_Scaling, 2, 1e3, 1e3],
                "Ackley6D":[Ackley6D, Ackley6D_Scaling, 6, 1e3, 1e3],
                "Ackley10D":[Ackley10D, Ackley10D_Scaling, 10, 1e3, 1e3],
                "Ashbychart":[Ashbychart_wrapper(datadir), Ashbychart_Scaling, 3, 1e3, 1e3], #Scaling incorporated into Ashbychart function directly
                "CantileverBeam":[CantileverBeam, CantileverBeam_Scaling, 10, 1e3, 1e3],
                "CompressionSpring":[CompressionSpring, CompressionSpring_Scaling, 3, 1e3, 1e3],
                "Car":[Car, Car_Scaling, 11, 1e3, 1e3],
                "HeatExchanger":[HeatExchanger, HeatExchanger_Scaling, 8, 1e3, 1e3],
                "PressureVessel":[PressureVessel, PressureVessel_Scaling, 4, 1e3, 1e3],
                "ReinforcedConcreteBeam":[ReinforcedConcreteBeam, ReinforcedConcreteBeam_Scaling, 3, 1e3, 1e3],
                "SpeedReducer":[SpeedReducer, SpeedReducer_Scaling, 7, 1e3, 1e3],
                "ThreeTruss":[ThreeTruss, ThreeTruss_Scaling, 2, 1e3, 1e3],
                "ToyCole1":[ToyCole1, ToyCole1_Scaling, 2, 1e3, 1e3],
                "ToyCole2":[ToyCole2, ToyCole2_Scaling, 2, 1e3, 1e3],
                "ToyGardner1":[ToyGardner1, ToyGardner1_Scaling, 2, 1e3, 1e3],
                "ToyGardner2":[ToyGardner2, ToyGardner2_Scaling, 2, 1e3, 1e3],
                "WeldedBeam":[WeldedBeam, WeldedBeam_Scaling, 4, 1e3, 1e3],
                }
    return function_dict[function_ID]

def generate_dataset_by_ID(function_ID, batch_size=10000, strong_negatives = False, datadir="./"):
    print(f"Generating dataset for function: {function_ID}")
    fn, scale_fn, dim, num_samples_p, num_samples_n = lookup_function(function_ID, datadir)
    if strong_negatives:
        num_samples_n = 10*strong_negatives
    num_samples_p = int(num_samples_p)
    num_samples_n = int(num_samples_n)
    lbs = scale_fn(torch.zeros(1, dim))
    ubs = scale_fn(torch.ones(1, dim))
    rangearr = np.concatenate([lbs, ubs]).T
    def validity_fn(X):
        X = torch.tensor(X)
        gx, fx = fn(X)
        gx = gx.numpy()
        return np.max(gx, axis=1) <= 0
    positives = []
    negatives = []
    while len(positives) < num_samples_p or len(negatives) < num_samples_n:
        print(len(positives), len(negatives))
        X = torch.rand(batch_size, dim)
        X_scaled = scale_fn(X)
        if isinstance(X_scaled, torch.Tensor):
            X_scaled = X_scaled.numpy()
        validity = validity_fn(X_scaled)
        positive_addition = X_scaled[validity]
        negative_addition = X_scaled[~validity]
        if positives!=[]:
            positives = np.concatenate((positives, positive_addition), axis=0)
        else:
            positives = positive_addition
        if negatives!=[]:
            negatives = np.concatenate((negatives, negative_addition), axis=0)
        else:
            negatives = negative_addition
        positives = positives[:num_samples_p]
        negatives = negatives[:num_samples_n]
    if strong_negatives:
        distances = scipy.spatial.distance.cdist(positives, negatives)
        min_distances = np.min(distances, axis=0)
        min_distance_idxs = np.argsort(min_distances)
        negatives = negatives[min_distance_idxs[:strong_negatives]]
        function_ID = function_ID + "_selected"
    np.save(f"{datadir}{function_ID}_distribution.npy", positives)
    np.save(f"{datadir}{function_ID}_negative.npy", negatives)
    np.save(f"{datadir}{function_ID}_rangearr.npy", rangearr)

def get_dataset_fn_by_ID(function_ID, datadir, strong_negatives=False):
    fn, scale_fn, dim, num_positive, num_negative = lookup_function(function_ID, datadir)
    if strong_negatives:
        function_ID = function_ID + "_selected"
    distribution = np.load(f"{datadir}{function_ID}_distribution.npy")
    negative = np.load(f"{datadir}{function_ID}_negative.npy")
    rangearr = np.load(f"{datadir}{function_ID}_rangearr.npy")
    def validity_fn(X):
        X = torch.tensor(X)
        gx, fx = fn(X)
        gx = gx.numpy()
        return np.max(gx, axis=1) <= 0
    def dataset_fn():
        return distribution, negative, None
    return dataset_fn, validity_fn, rangearr

