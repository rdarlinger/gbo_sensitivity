import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, pbdv
from scipy.optimize import curve_fit
import pandas as pd

def pdf_cosmic(DM, redshift, parameter_file):
    """
    Creates the p(DM|z) funciton using R.M. Konietzka et al. 2025 formulation
    
    Parameters:
    -------------
    DM: array
        DM values to calculate the pdf (with Milky Way contribution already subtracted off)
    redshift: array
        Redshift values to calculate the pdf
    parameter_file: string (csv file)
        File that contains sigma and mu values fit through a simulation to calculate mu and sigma at any redshift
        Should have columns of z, mu, and sigma
        
    Returns:
    -------------
    pdf_cosmic: array:shape(NDM,Nz)
        PDF of the given DM values
    """
    def poly(x,a,b,c,d,e):
        return a*x**4+b*x**3+c*x**2+d*x+e
    
    df = pd.read_csv(parameter_file)
    mus=df["mu"].values
    sigmas=df["sigma"].values
    z=df["z"].values
    
    sigma_fit=curve_fit(poly, z[1:], sigmas[1:], p0=[1,1,1,1,1])
    mu_fit=curve_fit(poly, z, mus, p0=[1,1,1,1,1])
    
    mu=poly(redshift, mu_fit[0][0], mu_fit[0][1], mu_fit[0][2], mu_fit[0][3], mu_fit[0][4]).reshape(1,-1)
    sigma=poly(redshift, sigma_fit[0][0], sigma_fit[0][1], sigma_fit[0][2], sigma_fit[0][3], sigma_fit[0][4]).reshape(1,-1)
    
    DM=DM[:, None]
    
    alpha=1
    beta=3.3
    eta=mu/(np.sqrt(2)*sigma*alpha)
    delta=(beta-1)/alpha
    log_c_inv=np.log(mu / alpha) -(eta**2)/2-delta*np.log(mu/(alpha*sigma))+np.log(gamma(delta))+np.log(pbdv(-delta, -np.sqrt(2)*eta)[0])

    pdf_cosmic= np.exp(-(mu**2*((mu/DM)**alpha-1)**2)/(2*sigma**2*alpha**2)-beta*np.log(DM/mu)-log_c_inv)
    
    return pdf_cosmic