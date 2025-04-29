# Host2D
An example implementation of Host2D model for SN Ia post-standardization residual (Murakami et al. 2025).
This model uses Scipy's LSQBivariateSpline to model the Hubble residuals over 2D surface (SN color -- host tracer space) to perform a continuous, color-dependent and host-dependent correction.

Please see ![host2d.ipynb](host2d.ipynb) for a step-by-step example.

A summary of the usage is below.
~~~python
from host2d import Host2DFitter

# assume you have a pandas dataframe that contains the necessary values.

### initialize the Host2DFitter object
host2d = Host2DFitter(
    color           = df['c'].values, 
    color_err       = df['cERR'].values, 
    tracer          = df['host_logmass'].values, 
    tracer_err      = df['host_logmass_err'].values,
    HR              = df['HR_tripp'].values,
    HR_err          = df['HR_tripp_err'].values,
    HR_err_nosigint = df['HR_tripp_err_nosigint'].values
)

### get data map
data_map,data_map_Ndata,domain_func = host2d.get_data_map(
    c_grid = np.linspace(-0.165, 0.3, 25), # SALT2 color grid
    tracer_grid = np.linspace(8, 11.4, 25), # global mass grid
    c_window = 0.05,
    tracer_window = 0.3,
    N_bootstrap = 3000,
    min_Ndata_required = 4
)

### fitting
host2d.prep_splint_fit(Nx=1,Ny=1,kx=1,ky=1) # set number of knots (Nx,Ny) and polynomial order (kx,ky)
spline = host2d.fit() # returns scipy.interpolate._fitpack2.LSQBivariateSpline object

### evaluate the result
HR_corr,HR_model,in_domain = host2d.evaluate() # uses the stored color and tracer values by default
chi2_original = np.sum(host2d.HR**2/host2d.HR_err**2)
chi2 = np.sum(HR_corr**2/host2d.HR_err**2)
print(f'Original chi2: {chi2_original:.2f}')
print(f'New chi2:      {chi2:.2f}')

host2d.plot_2D(ylabel='Host logmass')
host2d.plot_trace(tracer_label='Host\nLogmass')
~~~

![image](https://github.com/user-attachments/assets/2c156ae1-f7c8-4e60-9246-06fc800be3bf)
