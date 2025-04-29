import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import pandas as pd

from scipy.optimize import minimize
from scipy.interpolate import SmoothBivariateSpline, LSQBivariateSpline
from scipy.interpolate import RegularGridInterpolator

try:
    from p_tqdm import p_umap
except ImportError:
    print('p_tqdm not installed, using standard map instead')
    p_umap = map

def rolling_median_2D_MC(x_grid,y_grid,
                         xdata, xdata_err,
                         ydata, ydata_err,
                         zdata, zdata_err,
                         x_window, y_window,
                         N_bootstrap,
                         min_Ndata_required):
    
    def bootstrap_helper(i):
        ''' make a single bootstrap realization '''
        # Monte Carlo -- draw a random sample from the data
        x_sample = np.random.normal(xdata,xdata_err)
        y_sample = np.random.normal(ydata,ydata_err)
        z_sample = np.random.normal(zdata,zdata_err)
        
        # initialize the 2D maps
        z_binned = np.zeros((len(x_grid), len(y_grid)))*np.nan
        z_binned_Ndata = np.zeros((len(x_grid), len(y_grid)))*np.nan
        
        # fill the 2D maps
        for i,x_loc in enumerate(x_grid):
            for j,y_loc in enumerate(y_grid):
                _s =  (x_sample >= x_loc-x_window/2)
                _s &= (x_sample <  x_loc+x_window/2)
                _s &= (y_sample >= y_loc-y_window/2)
                _s &= (y_sample <  y_loc+y_window/2)
                z_binned_Ndata[i,j] = _s.sum()
                if _s.sum() < min_Ndata_required: 
                    continue
                z_binned[i,j] = np.median(z_sample[_s])
                
        return z_binned,z_binned_Ndata

    res = p_umap(bootstrap_helper,range(N_bootstrap))
    z_binned_all,z_binned_all_Ndata = zip(*res)
    
    # take the average of all the bootstrap realizations
    z_binned = np.nanmean(z_binned_all,axis=0)
    z_binned_Ndata = np.nanmean(z_binned_all_Ndata,axis=0)
    
    # generate the domain function for the map
    domain_func = get_domain_func(x_grid,y_grid,z_binned)
    return z_binned,z_binned_Ndata,domain_func

def get_domain_func(x_grid,y_grid,zz):
    ''' generates a function that returns true if a point is in the domain of the input map '''
    in_domain_func = RegularGridInterpolator(
        (x_grid, y_grid),
        np.isfinite(zz).astype(int),
        method='nearest',
        bounds_error=False,   # Points outside the grid will be handled by fill_value
        fill_value=0          # Out-of-bound observations are marked as 0 (False)
    )
    return in_domain_func

class Host2DFitter():
    def __init__(self,
                 color,color_err,
                 tracer,tracer_err,
                 HR,HR_err,HR_err_nosigint):
        self.color = color
        self.color_err = color_err
        self.tracer = tracer
        self.tracer_err = tracer_err
        self.HR = HR
        self.HR_err = HR_err
        self.HR_err_nosigint = HR_err_nosigint

    def get_data_map(self,c_grid,tracer_grid,c_window,tracer_window,
                     N_bootstrap=1000,min_Ndata_required=4):
        ''' Call rolling_median_2D_MC to get the data map. 
        Performs Monte Carlo and 2D rolling binning to get the 2D map of Hubble residuals over given grid values.
        
        Inputs:
            c_grid (np.array): grid of color values at which to evaluate the spline
            tracer_grid (np.array): grid of tracer values at which to evaluate the spline
            c_window (float): window size for color
            tracer_window (float): window size for tracer
            N_bootstrap (int): number of bootstrap realizations (default: 1000)
            min_Ndata_required (int): minimum number of data points required in each bin (default: 4)
            
        Outputs:
            HR_binned (np.array): the 2D HR values
            HR_binned_Ndata (np.array): number of data points in each bin
            domain_func (function): function that returns true if a point is in the domain of the map
        '''
        HR_binned,HR_binned_Ndata,domain_func = rolling_median_2D_MC(
            x_grid = c_grid,
            y_grid = tracer_grid,
            xdata = self.color,
            xdata_err = self.color_err,
            ydata = self.tracer,
            ydata_err = self.tracer_err,
            zdata = self.HR,
            zdata_err = self.HR_err_nosigint, # use the error without the intrinsic scatter
            x_window = c_window,
            y_window = tracer_window,
            N_bootstrap = N_bootstrap,
            min_Ndata_required = min_Ndata_required,
        )
        self.c_grid = c_grid
        self.tracer_grid = tracer_grid
        self.HR_binned = HR_binned
        self.HR_binned_Ndata = HR_binned_Ndata
        self.domain_func = domain_func        
        self.in_domain = domain_func(np.column_stack([self.color, self.tracer])) > 0 # True if in domain
        self.good_pix = np.isfinite(HR_binned.T.ravel())
        return HR_binned,HR_binned_Ndata,domain_func


    def prep_splint_fit(self,Nx,Ny,kx,ky,eps_range = 1e-4):
        ''' prepare bounds and initial positions for the spline fit '''
        # bounds 
        c_range = (self.c_grid.max() - self.c_grid.min()) * (1-eps_range) / Nx
        tracer_range = (self.tracer_grid.max() - self.tracer_grid.min()) * (1-eps_range) / Ny
        bounds_c = [(self.c_grid.min() + i*c_range + eps_range,
                     self.c_grid.min() + (i+1)*c_range) for i in range(Nx)]
        bounds_tracer = [(self.tracer_grid.min() + i*tracer_range + eps_range,
                          self.tracer_grid.min() + (i+1)*tracer_range) for i in range(Ny)]      
        
        # initial positions
        tx_init = np.linspace(self.c_grid.min(), self.c_grid.max(), Nx+1)[:-1] + c_range/2
        ty_init = np.linspace(self.tracer_grid.min(), self.tracer_grid.max(), Ny+1)[:-1] + tracer_range/2
        
        # flatten the grid, select in-domain values, and store them to be used during the fit
        self.xx, self.yy = np.meshgrid(self.c_grid, self.tracer_grid)
        self.xdata_fit = self.xx.ravel()[self.good_pix]
        self.ydata_fit = self.yy.ravel()[self.good_pix]
        self.zdata_fit = self.HR_binned.T.ravel()[self.good_pix]
        
        
        # store values
        self.Nx = Nx
        self.Ny = Ny
        self.kx = kx
        self.ky = ky
        self.init_knots = np.squeeze([*tx_init,*ty_init])
        self.bounds = [*bounds_c, *bounds_tracer] 
         
    def get_spline(self,theta):
        ''' use Scipy implementation to get the best-fit spline for the given knots '''
        if not hasattr(self,'Nx'):
            raise ValueError('Need to run prep_splint_fit() first')
        
        tx = theta[:self.Nx]
        ty = theta[self.Nx:]
        spline = LSQBivariateSpline(self.xdata_fit, 
                                    self.ydata_fit, 
                                    self.zdata_fit, 
                                    tx, ty, 
                                    kx=self.kx, ky=self.ky)
        return spline
    
    def knot_finding_helper(self,theta):
        ''' A wrapper function to evaluate the chi-square of the get_spline() result for given knots '''
        spline = self.get_spline(theta)
        model_vals = spline(self.color[self.in_domain],self.tracer[self.in_domain],grid=False)
        res = self.HR.copy()
        res[self.in_domain] -= model_vals
        chi2 = np.sum(res**2/self.HR_err**2)
        return chi2
    
    def fit(self,init_knots=None,
            method='Nelder-Mead',options={'fatol':1e-8,'xatol':1e-10,'maxfev':1e4},
            print_res=True):
        ''' Fit the spline to the data and optimize the knot location to minimize the residual.
        
        Inputs:
            init_knots: initial guess for the knot positions (default: self.init_knots)
            method: optimization method (default: 'Nelder-Mead')
            options: optimization options (default: {'fatol':1e-8,'xatol':1e-10,'maxfev':1e4})
            print_res: if True, print the optimization results (default: True)
            
        Outputs:
            self.spline: the best-fit spline (scipy.interpolate._fitpack2.LSQBivariateSpline object)
        '''
        if init_knots is None:
            init_knots = self.init_knots
        else:
            self.init_knots = init_knots
        
        # fit
        self.res = minimize(
            self.knot_finding_helper,
            init_knots,
            bounds=self.bounds,
            method=method,
            options=options
            )
        if print_res:
            print(self.res)
        self.tx_fit,self.ty_fit = self.res.x[:self.Nx],self.res.x[self.Nx:]
        self.spline = self.get_spline(self.res.x)

        self.HR_grid_spline = self.spline(self.c_grid,self.tracer_grid,grid=True)
        self.HR_grid_spline[~np.isfinite(self.HR_binned)] = np.nan
        self.free_parameters = len(self.spline.get_coeffs())
        return self.spline
    
    def plot_2D(self,vminmax=None,ylabel='tracer'):
        
        # define axes
        extent = [self.c_grid.min(),self.c_grid.max(),
                  self.tracer_grid.min(),self.tracer_grid.max()]
        aspect = (self.c_grid.max()-self.c_grid.min())/(self.tracer_grid.max()-self.tracer_grid.min())
        
        # color map range
        if vminmax is None:
            vmin,vmax = np.nanpercentile(self.HR_binned,[5,95])
        else:
            vmin,vmax = vminmax
            
        fig,axes = plt.subplots(1,2,figsize=(8,4))
        plt.subplots_adjust(wspace=0.05)

        axes[0].imshow(self.HR_binned.T, 
                       cmap='YlGnBu_r',origin='lower',
                       extent=extent,vmin=vmin,vmax=vmax)
        axes[0].set_aspect(aspect)
        axes[0].set_xlabel('SALT2 c',fontsize=13)
        axes[0].set_ylabel(ylabel,fontsize=13)
        axes[0].set_title('2D Rolling Median (data)',fontsize=13)

        im = axes[1].imshow(self.HR_grid_spline.T,
                            cmap='YlGnBu_r',origin='lower',
                            extent=extent,vmin=vmin,vmax=vmax)
        axes[1].set_aspect(aspect)
        axes[1].set_xlabel('SALT2 c',fontsize=13)
        axes[1].set_title(f'Bivariate Spline ({self.free_parameters}+2 parameters)',fontsize=13)
        for i in range(len(self.tx_fit)):
            for j in range(len(self.ty_fit)):
                axes[1].scatter(self.tx_fit[i],self.ty_fit[j],c='r')

        axes[1].set_yticklabels([])
        for ax in axes:
            ax.tick_params(direction='in',top=True,right=True)
        cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
        fig.colorbar(im, cax=cax, label='Residual', ax=axes.ravel().tolist())
        cax.set_ylabel(cax.get_ylabel(),fontsize=13)
        
    def plot_trace(self,tracer_label='tracer'):
        c_norm = Normalize(vmin=self.c_grid.min(), vmax=self.c_grid.max())
        tracer_norm = Normalize(vmin=self.tracer_grid.min(), vmax=self.tracer_grid.max())
        cmap1 = plt.get_cmap('RdYlGn_r')
        cmap2 = plt.get_cmap('cool')

        fig,axes = plt.subplots(2,2,figsize=(10,6))
        plt.subplots_adjust(hspace=0.05,wspace=0.25,top=0.9,bottom=0.1)
        cax1 = fig.add_axes([0.475,0.1,0.02,0.8])
        cax2 = fig.add_axes([0.905,0.1,0.02,0.8])

        for i in list(range(len(self.c_grid)))[::-1]:
            zorder = 100-((self.HR_binned_Ndata/self.HR_binned_Ndata.max()).T[i,:]*100).mean()
            axes[1,1].plot(self.xx[i,:],self.HR_grid_spline.T[i,:],
                           c=cmap1(tracer_norm(self.yy[i,:].mean())),lw=1.5)
            axes[0,1].scatter(self.xx[i,:],self.HR_binned.T[i,:],
                              fc=cmap1(tracer_norm(self.yy[i,:])),
                              s=(self.HR_binned_Ndata/self.HR_binned_Ndata.max()).T[i,:]*100+8,
                              ec='w',lw=0.1,zorder=zorder)

        for i in range(len(self.tracer_grid)):
            zorder = 100-((self.HR_binned_Ndata/self.HR_binned_Ndata.max()).T[:,i]*100).mean()
            axes[1,0].plot(self.yy[:,i],self.HR_grid_spline.T[:,i],
                           c=cmap2(c_norm(self.xx[:,i].mean())),lw=1.5)
            axes[0,0].scatter(self.yy[:,i],self.HR_binned.T[:,i],
                              fc=cmap2(c_norm(self.xx[:,i])),
                              s=(self.HR_binned_Ndata/self.HR_binned_Ndata.max()).T[:,i]*100+8,
                              ec='w',lw=0.1,zorder=zorder)

        for ax in axes[0,:]:
            ax.set_xticklabels([])
        for ax in axes[:,1]:
            ax.set_yticklabels([])
            
        axes[1,0].axvline(self.ty_fit,c='gray',ls=':',lw=2)
        axes[1,1].axvline(self.tx_fit,c='gray',ls=':',lw=2)
        axes[1,1].set_xlabel('SALT2 c',fontsize=13)
        axes[1,0].set_xlabel('Host Logmass',fontsize=13)
        axes[0,0].set_ylabel('Residual (data)',fontsize=13)
        axes[1,0].set_ylabel('Model: Host2D',fontsize=13)

        # axes[1].set_yticklabels([])
        for ax in axes.flatten():
            ax.set_ylim(axes[0,0].get_ylim())
            ax.axhline(0,c='k',ls='--',lw=0.5)
            ax.tick_params(direction='in',top=True,right=True)
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap2,norm=c_norm),cax=cax1)
        cax1.set_title('SALT2\nc',fontsize=10)
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap1,norm=tracer_norm),cax=cax2)
        cax2.set_title(tracer_label,fontsize=10)
        for ax in [cax1,cax2]:
            ax.tick_params(direction='in',labelsize=9)
            
            
    def evaluate(self,color=None,tracer=None,HR=None):
        ''' evaluate the spline at the given c and tracer values.
        
        Inputs:
            color: array of color values (default: self.color)
            tracer: array of tracer values (default: self.tracer)
            HR: array of HR values (default: self.HR)
            
        Ouputs:
            HR_corr: array of corrected HR values (HR_data - HR_model)
            HR_model: array of model HR values
            in_domain: boolean array indicating if the points are in the domain of the spline
        '''
        if not hasattr(self,'spline'):
            raise ValueError('Need to run fit() first')

        if color is None:
            color = self.color
        if tracer is None:
            tracer = self.tracer
        
        # check if c and tracer values are within domain
        in_domain = self.domain_func(np.column_stack([color, tracer])) > 0 
        
        # evaluate the spline
        HR_corr = self.HR.copy()
        HR_corr[in_domain] -= self.spline(color[in_domain],tracer[in_domain],grid=False)
        HR_model = self.spline(color,tracer,grid=False)
        
        return HR_corr,HR_model,in_domain