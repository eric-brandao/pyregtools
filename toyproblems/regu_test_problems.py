# -*- coding: utf-8 -*-
"""
some test problems from regu

@author: Admin
"""
import numpy as np
import scipy
import warnings
import matplotlib.pyplot as plt

def gravity_model(n,a,b,d):
    """ Defines the square matrix A by mid-point sampling
    
    Parameters:
    ------------
    n : int
        order of the model
    a : float
        bottom limit of the Fredholm integral
    b : float
        upper limit of the Fredholm integral
    d : float
        vertical distance from the source line (x') to measurement line (x)

    Returns:
    ------------
    A : numpy ndArray
        n x n real sensing matrix
    """
    dx = 1/n
    dxl = (b-a)/n
    x = dx * (np.arange(n) + 0.5) 
    xl = a + dxl * (np.arange(n) + 0.5) 
    X,XL = np.meshgrid(x,xl)
    A = dx * (d/(d**2 + (X-XL)**2)**(3/2)) 
    return A, x

def density_piecewise(n):
    """ Piecewise source function
    
    Defines a function of order n for which the first 1/3 of the source points are 2, and the rest are 1.
    
    Parameters:
    ------------
    n : int
        order of the model

    Returns:
    ------------
    rhox : numpy 1dArray
        pieciwise source term
    """
    nn = int(n/3)
    rhox = np.ones(n)
    rhox[:nn] = 2
    return rhox

def density_sin(x, fp):
    """ Sinusoidal source function
    
    Defines a function or order len(x) for a sinusoidal source function.
    
    Parameters:
    -----------
    x : 1dArray
        measured coordinates
    fp : float
        spatial frequency of the phenomenon
    Returns:
    ------------
    rhox : numpy 1dArray
        sinusoidal source term
    """
    rhox = np.sin(2*np.pi*fp*x)
    return rhox

def density_rechann(x, hann_x = 0.5):
    """ Rectangular + hanning density function
    
    Parameters:
    ------------
    n : int
        order of the model

    Returns:
    ------------
    rhox : numpy 1dArray
        pieciwise source term
    """
    n = len(x)
    nh = len(x[x>hann_x])
    hw = scipy.signal.windows.hann(2*nh)
    rhox = 2*np.ones(n)
    rhox[n-nh:] = hw[nh:]+1
    return rhox

class HeatInfRod(object):
    """ Heat on an infinite rod toy problem
    """
    def __init__(self, L=1.0, alpha = 1.0, tmax = 10, fs = 50):
        """ Init
        
        Parameters:
        ------------
        L : float
            downstream location where temperature measurement is taken
        alpha : float
            thermal diffusion coefficient of the rod
        tmax : 
            maximum measurement time in [s]
        fs : int
            sampling rate of the time measured signal
        """
        self.L = L
        self.alpha = alpha
        self.fs = fs
        self.time = np.arange(0, tmax, 1/fs) # time vector of measurement
    
    def heat_kernel(self, tau):
        """ Boundary Green's function for a semi-infinite rod.
                
        Returns
        -------
        K : ndarray
            Kernel evaluated at each time lag.
        """
        K = np.zeros_like(tau)
        mask = tau > 0
        K[mask] = (self.L/(2 * np.sqrt(np.pi * self.alpha) * tau[mask]**1.5))*\
            np.exp(-self.L**2 / (4 * self.alpha * tau[mask]))
        return K
    
    def heat_forward_matrix(self,):
        """ Construct the sensing matrix
        """
        dt = self.time[1] - self.time[0]
        n = len(self.time)
        self.G = np.zeros((n, n))
        for i in range(n):
            tau = self.time[i] - self.time[:i+1]
            self.G[i, :i+1] = self.heat_kernel(tau)
        self.G *= dt
        
    def heat_square_pulse(self, tinit = 1, tend = 2, Tsource = 200, Tbase = 0):
        """ Square pulse of temperature.
    
        Parameters
        ----------
        tinit : float
            initial time instant of pulse
        tend : float
            final time instant of pulse
        Tsource : float
            source temperature in K
        Tbase : float
            base temperature in K
        """
        self.u = np.zeros_like(self.time) + Tbase
        self.u[(self.time > tinit) & (self.time < tend)] = Tsource
        
    def heat_hann_pulse(self, tinit = 1, tend = 2, Tsource = 200, Tbase = 0):
        """ Hanning pulse of temperature.
    
        Parameters
        ----------
        tinit : float
            initial time instant of pulse
        tend : float
            final time instant of pulse
        Tsource : float
            source temperature in K
        Tbase : float
            base temperature in K
        """
        self.u = np.zeros_like(self.time) + Tbase
        dt = self.time[1] - self.time[0]
        start_index = int(tinit/dt)
        end_index = int(tend/dt)
        nh = end_index-start_index
        self.u[start_index:end_index] = (Tsource)*scipy.signal.windows.hann(nh)+Tbase

    def heat_random(self, mean_temp = 300, std_temp = 20):
        """ Random temperature fluctuation
        """
        self.u = np.random.normal(loc = mean_temp, scale = std_temp, size = len(self.time))
        
    def compute_temp(self):
        """ Compute true temperature at measurement location
        """
        self.temp_meas = self.G @ self.u
        
    def add_gaussian_noise(self, snr = 20, seed = 0):
        """ Add gaussian noise to true temperature
        """
        np.random.seed(seed)
        n = np.random.normal(loc = 0.0, scale = 1, size = len(self.temp_meas))
        n /= np.linalg.norm(n)
        n *= (10**(-snr/10))*np.linalg.norm(self.temp_meas)
        self.temp_meas += n

class Sky(object):
    """ Blurred Sky toy problem
    """
    def __init__(self, size_x = 20, size_y = 20):
        # self.size_x, self.size_y = size_x, size_y
        if size_x < 20 or size_y <20:
            warnings.warn("Size of sky should be at least 100 x 100 pixels")
            size_x, size_y = 20, 20
        # instance of empty sky
        self.sky = np.zeros((size_x, size_y))
        
    def add_star(self, pos = (10, 10), br = 1):
        """ adds a star at position "pos" with brightness br
        """
        self.sky[pos] = br
        
    def add_stars(self, pos = [(5, 5), (10,10)], br = [1, 0.8]):
        """ add several stars at once
        """
        for jp, p in enumerate(pos):
            self.add_star(pos = p, br = br[jp])

    def circular_psf(self, psf_radius = 1):
        """Generates a normalized circular (pillbox) PSF kernel.
        
        """
        self.psf_radius = psf_radius
        size = int(2*self.psf_radius + 1)
        y, x = np.ogrid[-self.psf_radius:self.psf_radius+1, -self.psf_radius:self.psf_radius+1]
        # Create a binary mask where distance from center is <= radius
        mask = (x**2 + y**2 <= self.psf_radius**2)
        # Initialize kernel and normalize so all elements sum up to 1.0
        self.kernel = np.zeros((size, size))
        self.kernel[mask] = 1.0
        self.kernel /= self.kernel.sum()
        
    def blur_sky_im(self, psf_radius = 1):
        """ blur the sky image
        """
        self.circular_psf(psf_radius = psf_radius)
        self.sky_blurred = scipy.signal.convolve2d(self.sky, self.kernel, mode='same')
        
    def add_gaussian_noise(self, snr = 20, seed = 0):
        """ Add gaussian noise to blurred sky
        """
        np.random.seed(seed)
        n = np.random.normal(loc = 0.0, scale = 1, size = len(self.sky_blurred.flatten()))
        n /= np.linalg.norm(n)
        n *= (10**(-snr/10))*np.linalg.norm(self.sky_blurred.flatten())
        self.sky_blurred += np.reshape(n, self.sky.shape)
        
    def add_poisson_noise(self, snr = 20, seed = 0):
        """
        Scales a blurred image to hit a target peak SNR under Poisson noise,
        applies the noise, and returns the image scaled back to original units.
        """
        # Enforce non-negativity
        clean_img = np.clip(self.sky_blurred, 0, None)
        # Find the current maximum pixel value in the blurred image
        current_peak = np.max(clean_img)
        # Calculate the required target peak photon count
        # Since SNR = sqrt(Intensity), Target_Intensity = SNR^2
        target_peak_value = snr ** 2
        # Calculate the scaling (exposure) factor
        scale_factor = target_peak_value / current_peak
        # Scale image to photon counts
        photon_image = clean_img * scale_factor
        # Apply Poisson noise
        noisy_photon_image = np.random.poisson(photon_image)
        # 7. Scale back to original intensity units
        self.sky_blurred = noisy_photon_image / scale_factor
            
    def get_blur_matrix(self,):
        """ Get sensing matrix
        
        Creates an explicit sparse blur matrix H for a given 2D PSF kernel.
        Assumes 'same' boundary conditions with zero padding.
        """
        H_rows, H_cols = self.sky.shape
        N = H_rows * H_cols  # Total number of pixels
        # Get coordinates of the PSF relative to its center
        kh, kw = self.kernel.shape
        kh2, kw2 = kh // 2, kw // 2
        
        # We will build the sparse matrix using the 'LIL' (List of Lists) format for speed
        self.A = scipy.sparse.lil_matrix((N, N))
        
        # Loop through every pixel in the 2D image
        for r in range(H_rows):
            for c in range(H_cols):
                row_idx = r * H_cols + c  # Flattened 1D index of the current pixel
                # Place the PSF weights on the neighboring pixels
                for kr in range(kh):
                    for kc in range(kw):
                        weight = self.kernel[kr, kc]
                        if weight == 0:
                            continue
                        # Target pixel coordinates in the image
                        target_r = r + (kr - kh2)
                        target_c = c + (kc - kw2)
                        # Check boundary conditions (Zero Padding)
                        if 0 <= target_r < H_rows and 0 <= target_c < H_cols:
                            col_idx = target_r * H_cols + target_c
                            self.A[row_idx, col_idx] = weight     
        self.A.tocsr() # Convert to CSR format for fast math operations
        # return H.tocsr() # Convert to CSR format for fast math operations
    
    def pixel_sum(self,):
        """ print pixel sum
        """
        print(r"True sky brightness sum: B = {:.3f}".format(np.sum(self.sky)))
        print(r"Blurred sky brightness sum: B = {:.3f}".format(np.sum(self.sky_blurred)))
        
    def check_sens_mtx(self,):
        """ Check sensing matrix forward problem vs. conv
        """
        sb = self.A @ self.sky.flatten() # forward problem
        sb = np.reshape(sb, self.sky.shape)
        norm = np.linalg.norm(sb-self.sky_blurred)
        print("Norm difference (forward - conv) is {:.16f}".format(norm))

    def plot_true_sky(self):
        """ plots true sky
        """
        self.plot_sky(self.sky)
    
    def plot_blurred_sky(self):
        """ plots true sky
        """
        self.plot_sky(self.sky_blurred)

    def plot_sky(self, sky):
        """ plots true sky
        """
        plt.figure(figsize = (4,4))
        plt.imshow(sky.T, cmap='grey', origin='lower')
        plt.colorbar(shrink = 0.5, label = 'brightness')
        plt.axis("off") 
        plt.tight_layout()
        
class SoundField2D(object):
    """ 2D sound field estimation with plane waves
    """
    def __init__(self, c0 = 340):
        self.c0 = c0
    
    def add_mic_array(self, x_len = 0.5, z_len = 0.5, n_x = 5, n_z =5):
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        zc = np.linspace(-z_len/2, z_len/2, n_z)
        # meshgrid
        self.x_grid, self.z_grid = np.meshgrid(xc, zc)
        # initialize receiver list in memory
        self.coord = np.zeros((n_x * n_z, 2))
        self.coord[:, 0] = self.x_grid.flatten()
        self.coord[:, 1] = self.z_grid.flatten()
    
    def get_pw(self, theta_deg = 90):
        """ Adds a plane wave
        """
        kx = self.k0*np.sin(np.deg2rad(theta_deg))
        kz = self.k0*np.cos(np.deg2rad(theta_deg))
        return np.array([kx,kz])
    
    def compute_pres(self, freq = 1000, theta_deg = [45, -45], amps = [1, 0.7]):
        """Computes sound pressure for a set of plane waves
        """
        self.k0 = 2*np.pi*freq/self.c0
        self.pres = np.zeros(self.coord.shape[0], dtype = complex)
        for jt, theta in enumerate(theta_deg):
            k_vec = self.get_pw(theta_deg = theta)
            self.pres += amps[jt]*np.exp(-1j*self.coord @ k_vec)
            
    def add_noise(self, snr = 30, seed = 0):
        """ Add gaussian noise to the simulated data.

        """
        signalPower_lin = (np.mean(np.abs(self.pres))/np.sqrt(2))**2
        signalPower_dB = 10 * np.log10(signalPower_lin)
        noisePower_dB = signalPower_dB - snr
        noisePower_lin = 10 ** (noisePower_dB/10)
        np.random.seed(seed)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = self.pres.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = self.pres.shape)
        self.pres += noise
        
    def wave_directions(self,):
        """ Get plane-wave directions
        """
        delta_theta = 2*np.pi/self.nwaves
        theta_dir = np.arange(0, 2*np.pi, delta_theta)
        return theta_dir
    
    def get_sens_mtx(self, nwaves = 180):
        """ Get sensing matrix
        """
        self.nwaves = nwaves
        theta_dir = self.wave_directions()
        k_sens_vec = self.k0 * np.array([np.sin(theta_dir), np.cos(theta_dir)])
        self.A = np.exp(-1j * self.coord @ k_sens_vec)                
            
    def plot_measured_field(self):
        """ Plots measured sound field
        """
        min_val, max_val = np.real(self.pres).min(), np.real(self.pres).max()
        range_c = np.amax(np.abs([min_val, max_val]))
        plt.figure(figsize=(4, 4))
        scatter = plt.scatter(self.coord[:,0], self.coord[:,1], 
                              c=np.real(self.pres), cmap='bwr', s=30, edgecolor='none',
                              vmin = -range_c, vmax = range_c)
        cbar = plt.colorbar(scatter)
        cbar.set_label(r'$\mathcal{R}\left\{p\right\}$', fontsize=10)
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$z$ [m]")
        plt.tight_layout()
            
    def plot_sol_field(self, x_sol, normalize = True):
        """ Plots solution sound field
        """
        if normalize:
            x_sol /= np.amax(np.abs(x_sol))
        # scatter - coord
        theta_dir = self.wave_directions()
        coords_sol = np.array([np.sin(theta_dir), np.cos(theta_dir)]).T
        
        plt.figure(figsize=(5, 4))
        scatter = plt.scatter(coords_sol[:,0], coords_sol[:,1], 
                              c=np.abs(x_sol), cmap='inferno', s=30, edgecolor='none',
                              vmin = 0, vmax = np.abs(x_sol).max())
        cbar = plt.colorbar(scatter)
        cbar.set_label(r'$|x|$', fontsize=10)
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$z$ [m]")
        plt.tight_layout()
        
    def plot_sol_xy(self, x_sol):
        theta_dir = self.wave_directions()
        plt.figure(figsize=(6, 3))
        plt.plot(np.rad2deg(theta_dir), np.abs(x_sol), 'k')
        plt.grid(linestyle = '--')
        plt.xlim((0, 360))
        plt.ylim((-0.10, 1.2*np.abs(x_sol).max()))
        plt.xlabel(r"$\theta$ [deg]")
        plt.ylabel(r"$|x|$ [Pa]")
        plt.tight_layout()
        

class MultiDecay(object):
    """ multiple exponential decay problem

    """
    def __init__(self, fs = 10000, tmax = 5, amps = [1, 0.25, 0.0001],
                 t_decs = [1, 0.5]):
        """ init method
            
        Parameters
        ----------
        fs : int
            sampling rate of the time measured signal
        tmax : 
            maximum measurement time in [s]
        """
        self.fs = fs   
        self.fs = fs
        self.time = np.arange(0, tmax, 1/fs) # time vector of measurement
        self.amps = amps
        self.t_decs = t_decs
    
    def single_decay(self, amp = 1, t_dec = 1):
        """ computes a given exponential decay
        
        Parameters
        ----------
        amp : float
            amplitude of decay
        t_dec : float
            decay time in [s]
        """
        decay = amp*np.exp(-13.8*self.time/t_dec)
        return decay
        
    def noise_term(self, amp):
        """ noise term
        
        Parameters
        ----------
        amp : float
            noise amplitude
        """
        noise = amp*(self.time[-1]-self.time)
        return noise
    
    def compose_decay(self):
        """ compose decays
        """
        self.edc = np.zeros(len(self.time))
        for jd, td in enumerate(self.t_decs):
            self.edc += self.single_decay(amp = self.amps[jd], t_dec = td)
        self.edc += self.noise_term(amp = self.amps[jd+1])
        
    def add_gaussian_noise(self, snr = 20, seed = 0):
        """ Add gaussian noise to true temperature
        """
        np.random.seed(seed)
        noise_amp = 10**(-snr/10)
        n = np.random.normal(loc = 0.0, scale = noise_amp*self.edc, size = len(self.edc))
        self.edc += n
        
    def sensing_mtx(self, t_dec_range = [0.05, 3], order = 30):
        """ Build sensing matrix
        """
        t_decs = np.linspace(t_dec_range[0], t_dec_range[1], order)
        sens_mtx = np.zeros((len(self.time), len(t_decs)+1))
        for jd, td in enumerate(t_decs):
            sens_mtx[:, jd] = self.single_decay(amp = 1, t_dec = td)
        sens_mtx[:, -1] = self.time[-1]-self.time
        return sens_mtx        
        
    def plot_edc(self):
        """ plots the energy decay curve in dB
        """
        plt.figure(figsize = (6,4))
        plt.plot(self.time, 10*np.log10(self.edc/np.amax(self.edc)), 'k',
                 linewidth = 1.5, label = "True EDC")
        for jd, td in enumerate(self.t_decs):
            edc = self.single_decay(amp = self.amps[jd], t_dec = td)
            plt.plot(self.time, 10*np.log10(edc), '--', 
                     color = 'grey', label = "Decay {}".format(jd))
        noise = self.noise_term(amp = self.amps[jd+1])
        plt.plot(self.time[:-2], 10*np.log10(noise[:-2]), ':', 
                 color = 'grey', label = "Noise")
        plt.legend()
        plt.grid(linestyle = '--')
        plt.xlim([0, self.time[-1]])
        plt.ylim([-60, 2])
        plt.xlabel("Time [s]")
        plt.ylabel("EDC [dB]")
        plt.tight_layout()

            
        
