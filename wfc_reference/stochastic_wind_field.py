# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import pandas as pd
import scipy.interpolate as interp


class stochastic_wind_directions():
    def __init__(self,
                 coords_x,
                 coords_y,
                 t_len=3600.,
                 dt=1.,
                 wd_std = 11.,
                 wd_mean=270.,
                 alpha_lat=1.0,
                 alpha_lon=0.5,
                 include_low_freq_wdirs=True,
                 f_c=0.001,
                 n_ord=1,
                 shift_time=False,
                 ws_mean=8.,
                 seed=None
                 ):

        """
        Instantiates a stochastic_wind_directions object used to create 
        correlated stochastic wind direction time series at multiple turbine
        locations.

        Args:
            coords_x (list or np.array): List of wind turbine x (Easting) 
                coordinates (m).
            coords_y (list or np.array): List of wind turbine y (Northing) 
                coordinates (m).
            t_len (float): Length of wind direction time series (s).
            dt (float): Time step (s). NOTE: t_len/dt must be an even number!
            wd_std (float): Standard deviation of high frequency wind 
                direction (deg). The default value of 11 degrees is based on 
                1-hour wind direction std. dev. for neutral conditions measured 
                at the M5 met tower at the NWTC.
            wd_mean (float): Mean wind direction (deg).
            alpha_lat (float): Exponential decay constant for spatial coherence
                in lateral (cross-wind) direction.
            alpha_lon (float): Exponential decay constant for spatial coherence
                in longitudinal (along-wind) direction.
            include_low_freq_wdirs (bool): If True, low frequency wind direction 
                time series corresponding to each high frequency wind direction 
                time series will be generated. If False, only high freqiency 
                wind directions will be generated.
            f_c (float): Cutoff frequency for determining low frequency wind 
                direction (Hz).
            n_ord (float): Butterworth LPF filter order for determining low 
                frequency wind direction).
            shift_time (bool): If True, Wind direction time series will be 
                shifted according to the propogation delay determined from the 
                mean wind speed. If False, all wind direction time series will 
                be in phase.
            ws_mean (float): Mean wind speed. Only used if shift_time is 
                True (m/s).
            seed (int): If defined, used to initialize random number generator.
        """

        self.reinitialize_stochastic_wind_directions(
                 coords_x=coords_x,
                 coords_y=coords_y,
                 t_len=t_len,
                 dt=dt,
                 wd_std=wd_std,
                 wd_mean=wd_mean,
                 alpha_lat=alpha_lat,
                 alpha_lon=alpha_lon,
                 include_low_freq_wdirs=include_low_freq_wdirs,
                 f_c=f_c,
                 n_ord=n_ord,
                 shift_time=shift_time,
                 ws_mean=ws_mean,
                 seed=seed
                 )

    def _wind_direction_psd_simple(self):
        """ 
        Set high frequency PSD to simple 1/f function. Based on observations
        from M5 met tower at NWTC. 
        """
        return 1./self.freqs

    def _butterworth_lpf_mag(self):
        """
        Returns the magnitude of Butterworth filter transfer function 
        with cutoff frequency f_c and order n_ord.
        """
        return 1./np.sqrt(1.+(self.freqs/self.f_c)**(2*self.n_ord))

    def _exponential_coherence_combined(self,freq):
        """
        Returns spatial coherence betwen all turbine locations for a given frequency 
        using a simple exponential decay model for spatial separation r_lat and 
        decay constant alpha_lat for the lateral (cross-wind) direction and spatial 
        separation r_lon and decay constant alpha_lon for the longitudinal 
        (along-wind) direction. For longitudinal wind direction coherence, 
        alpha = 0.5 to 1 is a reasonable estimate based on analysis of SCADA 
        data from the Lillgrund wind farm. 
        """
        return np.exp(
                -np.sqrt((self.alpha_lon*self.delta_x)**2 + (self.alpha_lat*self.delta_y)**2)
                * freq
            )

    def _create_two_sided_freqs(self,freq_mat):
        """
        Convert the correlated one sided frequency vectors for each turbine to 
        two-sided frequency vectors.
        """
        freq_mat_twoside = np.zeros((self.num_turbs,self.num_samples_tot),dtype=complex)
        freq_mat_twoside[:,1:int(self.num_samples_tot/2)+1] = freq_mat
        freq_mat_twoside[:,int(self.num_samples_tot/2)+1:] = np.conj(np.flip(freq_mat[:,:-1],1))

        return freq_mat_twoside

    def _apply_freq_magnitudes_twoside(self,freq_vec,psd_oneside):
        """
        Apply one-sided frequency domain magnitudes from PSD to two-sided 
        frequency vector.
        """
        freq_vec_scaled = freq_vec.copy()
        freq_vec_scaled[1:int(self.num_samples_tot/2)+1] *= np.sqrt(psd_oneside)
        freq_vec_scaled[int(self.num_samples_tot/2)+1:] *= np.flip(np.sqrt(psd_oneside[:-1]))

        return freq_vec_scaled

    def reinitialize_stochastic_wind_directions(self,
                 coords_x=None,
                 coords_y=None,
                 t_len=None,
                 dt=None,
                 wd_std=None,
                 wd_mean=None,
                 alpha_lat=None,
                 alpha_lon=None,
                 include_low_freq_wdirs=None,
                 f_c=None,
                 n_ord=None,
                 shift_time=None,
                 ws_mean=None,
                 seed=None
                 ):

        """
        This method reinitializes any stochastic wind direction generator 
        parameters that are specified. Otherwise, the current parameters are kept.

        Args:
            coords_x (list or np.array): List of wind turbine x (Easting) 
                coordinates (m).
            coords_y (list or np.array): List of wind turbine y (Northing) 
                coordinates (m).
            t_len (float): Length of wind direction time series (s).
            dt (float): Time step (s). 
            wd_std (float): Standard deviation of high frequency wind 
                direction (deg).
            wd_mean (float): Mean wind direction (deg).
            alpha_lat (float): Exponential decay constant for spatial coherence
                in lateral (cross-wind) direction.
            alpha_lon (float): Exponential decay constant for spatial coherence
                in longitudinal (along-wind) direction.
            include_low_freq_wdirs (bool): If True, low frequency wind direction 
                time series corresponding to each high frequency wind direction 
                time series will be generated. If False, only high freqiency 
                wind directions will be generated.
            f_c (float): Cutoff frequency for determining low frequency wind 
                direction (Hz).
            n_ord (float): Butterworth LPF filter order for determining low 
                frequency wind direction).
            shift_time (bool): If True, Wind direction time series will be 
                shifted according to the propogation delay determined from the 
                mean wind speed. If False, all wind direction time series will 
                be in phase.
            ws_mean (float): Mean wind speed. Only used if shift_time is 
                True (m/s).
            seed (int): If defined, used to initialize random number generator.
        """

        if coords_x is not None:
            self.coords_x = np.array(coords_x)

            # Update number of turbines
            self.num_turbs = len(self.coords_x)
        if coords_y is not None:
            self.coords_y = np.array(coords_y)
        if t_len is not None:
            self.t_len = t_len
        if dt is not None:
            self.dt = dt
        if wd_std is not None:
            self.wd_std = wd_std
        if wd_mean is not None:
            self.wd_mean = wd_mean
        if alpha_lat is not None:
            self.alpha_lat = alpha_lat
        if alpha_lon is not None:
            self.alpha_lon = alpha_lon
        if include_low_freq_wdirs is not None:
            self.include_low_freq_wdirs = include_low_freq_wdirs
        if f_c is not None:
            self.f_c = f_c
        if n_ord is not None:
            self.n_ord = n_ord
        if shift_time is not None:
            self.shift_time = shift_time
        if ws_mean is not None:
            self.ws_mean = ws_mean
        if seed is not None:
            # Initialize random number generator
            np.random.seed(seed)
        

        if (t_len is not None) | (dt is not None):
            # Update number of samples in time series
            self.num_samples = int(self.t_len/self.dt)

        if (wd_mean is not None) | (coords_x is not None) | (coords_y is not None):
            # Rotate coordinates so new x axis is aligned with wind direction
            self.coords_x_rot = (np.cos(np.radians(self.wd_mean - 270.))*self.coords_x 
                    - np.sin(np.radians(self.wd_mean - 270.))*self.coords_y
                )

            self.coords_y_rot = (np.sin(np.radians(self.wd_mean - 270.))*self.coords_x 
                    + np.cos(np.radians(self.wd_mean - 270.))*self.coords_y
                )

            # Initialize inter-turbine distance matrices used to calculate coherence
            self.delta_x = np.abs(
                    np.matmul(np.transpose(np.array([self.coords_x_rot])),np.ones((1,self.num_turbs))) 
                    - np.matmul(np.ones((self.num_turbs,1)),np.array([self.coords_x_rot]))
                )
            self.delta_y = np.abs(
                    np.matmul(np.transpose(np.array([self.coords_y_rot])),np.ones((1,self.num_turbs))) 
                    - np.matmul(np.ones((self.num_turbs,1)),np.array([self.coords_y_rot]))
                )

        # If time shifts are used, add additional time samples to get the 
        # desired time series length. Always recalculate since this depends on
        # many parameters. TODO: Check for relevant changed variables before 
        # executing rest of code to save time (e.g., if only seed is not None, 
        # don't need to recalculate anything else)?
        self.num_samples_tot = self.num_samples

        if self.shift_time:
            self.time_delays = np.round((np.max(self.coords_x_rot) - self.coords_x_rot)/self.ws_mean).astype(int)
            self.num_samples_tot = self.num_samples_tot + int(2*np.ceil(np.max(self.time_delays)/2))

        # Create one-sided frequency vector
        self.freqs = np.arange(1/self.num_samples_tot,0.5+1/self.num_samples_tot,1/self.num_samples_tot)/self.dt

        # Initialize one-sided wind direction power spectra
        self.psd_oneside = self._wind_direction_psd_simple()
        self.psd_lowfreq_oneside = self.psd_oneside*(self._butterworth_lpf_mag()**2)

    def generate_timeseries(self):
        """
        Generates correlated stochastic high frequency and low frequency wind
        direction time series at multiple turbine locations based on the Veers 
        method used in TurbSim. The power spectrum of the high freqiency wind 
        directions, meant to represent wind vane maesurements, is given by a 
        simple 1/f function. Spatial coherence is defined using an exponential 
        decay coherence model. The low frequency wind direction time series, 
        meant to be used as FLORIS inputs, are formed by applying a the 
        magnitude of a Butterworth LPF transfer function to the high freqiency 
        wind directions. 

        Returns:
            pandas.DataFrame: A dataframe high frequency and low frequency wind 
            direction time series for each turbine location, along with a time column.
        """
        
        # To store Fourier components for each turbine location
        freq_mat = np.zeros((self.num_turbs,len(self.freqs)),dtype=complex)

        # Correlate phases for each frequency component
        for i_f in range(len(self.freqs)):

            # sqrt of coherence matrix
            G = np.sqrt(self._exponential_coherence_combined(self.freqs[i_f]))

            # Cholesky decomposition
            L = np.linalg.cholesky(G)
            
            # Correlate phases
            phase_vec = np.exp(np.random.uniform(high=2*np.pi,size=self.num_turbs)*1.0j)
            freq_mat[:,i_f] = np.matmul(L,phase_vec)
        
        # create two-sided frequency matrix
        freq_mat_twoside = self._create_two_sided_freqs(freq_mat)

        # Store wind direction time series for each turbine location
        wdirs = np.zeros((self.num_turbs,self.num_samples_tot))
        wdirs_lowfreq = np.zeros((self.num_turbs,self.num_samples_tot))

        # Generate time series for each turbine location
        for i in range(self.num_turbs):

            # apply high-frequency PSD magnitudes to frequency components 
            freq_vec_orig = freq_mat_twoside[i,:]
            freq_vec = self._apply_freq_magnitudes_twoside(freq_vec_orig,self.psd_oneside)
            
            # determine scaling factor to get standard deviation of 1
            if i == 0:
                scale_const_total = self.num_samples_tot/np.sqrt(np.sum(np.abs(freq_vec)**2))
            
             # same scaling for all turbine locations
            freq_vec = scale_const_total*freq_vec

            # create time series of high-frequency wind direction
            wdirs[i,:] = (self.wd_std*np.real(np.fft.ifft(freq_vec))+self.wd_mean) % 360

            # apply low-frequency PSD magnitudes to frequency components
            # freq_vec = freq_mat_twoside[i,:]
            freq_vec = self._apply_freq_magnitudes_twoside(freq_vec_orig,self.psd_lowfreq_oneside)
            
            # same scaling for all turbine locations
            freq_vec = scale_const_total*freq_vec

            # create time series of low-frequency wind direction
            wdirs_lowfreq[i,:] = (self.wd_std*np.real(np.fft.ifft(freq_vec))+self.wd_mean) % 360

        # Apply time shifts to account for advection time downstream
        if self.shift_time:
            for i in range(self.num_turbs):
                wdirs[i,:self.num_samples] = (
                    wdirs[i,self.time_delays[i]:self.time_delays[i] + self.num_samples]
                )
                wdirs_lowfreq[i,:self.num_samples] = (
                    wdirs_lowfreq[i,self.time_delays[i]:self.time_delays[i] + self.num_samples]
                )
        wdirs = wdirs[:,:self.num_samples]
        wdirs_lowfreq = wdirs_lowfreq[:,:self.num_samples]

        # Save as dataframe
        if self.include_low_freq_wdirs:
            df_hf = pd.DataFrame(wdirs.T,columns=['wd_%03d' % i for i in range(self.num_turbs)])
            df_lf = pd.DataFrame(wdirs_lowfreq.T,columns=['wd_lowfreq_%03d' % i for i in range(self.num_turbs)])
            df = pd.concat([df_hf,df_lf],1)
        else:
            df = pd.DataFrame(wdirs.T,columns=['wd_%03d' % i for i in range(self.num_turbs)])

        # Create time column
        df['time'] = np.arange(0.,self.t_len,self.dt)

        return df


class constrained_stochastic_wind_directions(stochastic_wind_directions):
    def __init__(self,
                 coords_x,
                 coords_y,
                 wds_constr,
                 inds_constr,
                 dt=1.,
                 wd_mean=270.,
                 alpha_lat=1.0,
                 alpha_lon=0.5,
                 include_low_freq_wdirs=True,
                 f_c=0.001,
                 n_ord=1,
                 shift_time=False,
                 ws_mean=8.,
                 seed=None
                 ):

        """
        Instantiates a constrained_stochastic_wind_directions object used to 
        create correlated stochastic wind direction time series at multiple 
        turbine locations constrained to have the provided wind direction time 
        series at specific locations.

        Args:
            coords_x (list or np.array): List of wind turbine x (Easting) 
                coordinates (m).
            coords_y (list or np.array): List of wind turbine y (Northing) 
                coordinates (m).
            wds_constr (np.array or np.ndarray): A single time series or an 
                MxN array of M time series of length N which act as constraints 
                at specific locations. 
            inds_constr (list): Indices of coordinates for which wind 
                direction time series constraints are provided.
            dt (float): Time step (s). NOTE: the time series length/dt must be 
                an even number!
            wd_mean (float): Mean wind direction (deg).
            alpha_lat (float): Exponential decay constant for spatial coherence
                in lateral (cross-wind) direction.
            alpha_lon (float): Exponential decay constant for spatial coherence
                in longitudinal (along-wind) direction.
            include_low_freq_wdirs (bool): If True, low frequency wind direction 
                time series corresponding to each high frequency wind direction 
                time series will be generated. If False, only high freqiency 
                wind directions will be generated.
            f_c (float): Cutoff frequency for determining low frequency wind 
                direction (Hz).
            n_ord (float): Butterworth LPF filter order for determining low 
                frequency wind direction).
            shift_time (bool): If True, Wind direction time series will be 
                shifted according to the propogation delay determined from the 
                mean wind speed. If False, all wind direction time series will 
                be in phase.
            ws_mean (float): Mean wind speed. Only used if shift_time is 
                True (m/s).
            seed (int): If defined, used to initialize random number generator.
        """

        self.reinitialize_constrained_stochastic_wind_directions(
                 coords_x=coords_x,
                 coords_y=coords_y,
                 wds_constr=wds_constr,
                 inds_constr=inds_constr,
                 dt=dt,
                 wd_mean=wd_mean,
                 alpha_lat=alpha_lat,
                 alpha_lon=alpha_lon,
                 include_low_freq_wdirs=include_low_freq_wdirs,
                 f_c=f_c,
                 n_ord=n_ord,
                 shift_time=shift_time,
                 ws_mean=ws_mean,
                 seed=seed
                 )

    def _interpolate_psds(self):
        """
        Interpolate to find the PSDs at all turbine locations based on the PSDs 
        of the constraint time series. For locations outside of a convex hull 
        formed by the constraint locations, nearest neighbor interpolation will 
        be used. 
        
        TODO: Improve the interpolation by applying more linear interpolation outside of the region encompassed by the 
        constraint locations, rather than only using nearest neighbor.
        """
        self.psd_mat = np.zeros((self.num_turbs,len(self.freqs)))

        if self.num_constr == 1:
            self.psd_mat = np.ones((self.num_turbs,1))*self.psd_mat_constr[0,:]
        elif self.num_constr == 2:
            interp_nearest = interp.NearestNDInterpolator(
                list(zip(self.coords_x[:self.num_constr],self.coords_y[:self.num_constr])),
                self.psd_mat_constr
                )
            self.psd_mat = interp_nearest(list(zip(self.coords_x,self.coords_y)))
        else:
            interp_nearest = interp.NearestNDInterpolator(
                list(zip(self.coords_x[:self.num_constr],self.coords_y[:self.num_constr])),
                self.psd_mat_constr
                )

            try:
                # First see if a valid linear interpolator can be created given 
                # the constraint coordinates
                interp_lin = interp.LinearNDInterpolator(
                    list(zip(self.coords_x[:self.num_constr],self.coords_y[:self.num_constr])),
                    self.psd_mat_constr
                    )
                
                self.psd_mat = interp_lin(list(zip(self.coords_x,self.coords_y)))
            except:
                # Otherwise, resort to nearest neighbor interpolation
                self.psd_mat = interp_nearest(list(zip(self.coords_x,self.coords_y)))
            else:
                # Even if linear interpolation is valid, identify locations where it 
                # didn't work and use nearest neighbor interpolation
                idx = np.where(np.isnan(self.psd_mat[:,0]))[0]

                if len(idx) > 0:
                    self.psd_mat[idx,:] = interp_nearest(
                        list(zip([self.coords_x[i] for i in idx],[self.coords_y[i] for i in idx]))
                        )


    def reinitialize_constrained_stochastic_wind_directions(self,
                 coords_x=None,
                 coords_y=None,
                 wds_constr=None,
                 inds_constr=None,
                 dt=None,
                 wd_mean=None,
                 alpha_lat=None,
                 alpha_lon=None,
                 include_low_freq_wdirs=None,
                 f_c=None,
                 n_ord=None,
                 shift_time=None,
                 ws_mean=None,
                 seed=None
                 ):

        """
        This method reinitializes any constrained stochastic wind direction generator 
        parameters that are specified. Otherwise, the current parameters are kept.

        Args:
            coords_x (list or np.array): List of wind turbine x (Easting) 
                coordinates (m).
            coords_y (list or np.array): List of wind turbine y (Northing) 
                coordinates (m).
            wds_constr (np.array or np.ndarray): A single time series or an 
                MxN array of M time series of length N which act as constraints 
                at specific locations. 
            inds_constr (list): Indices of coordinates for which wind 
                direction time series constraints are provided.
            dt (float): Time step (s). NOTE: the time series length/dt must be 
                an even number!
            wd_mean (float): Mean wind direction (deg).
            alpha_lat (float): Exponential decay constant for spatial coherence
                in lateral (cross-wind) direction.
            alpha_lon (float): Exponential decay constant for spatial coherence
                in longitudinal (along-wind) direction.
            include_low_freq_wdirs (bool): If True, low frequency wind direction 
                time series corresponding to each high frequency wind direction 
                time series will be generated. If False, only high freqiency 
                wind directions will be generated.
            f_c (float): Cutoff frequency for determining low frequency wind 
                direction (Hz).
            n_ord (float): Butterworth LPF filter order for determining low 
                frequency wind direction).
            shift_time (bool): If True, Wind direction time series will be 
                shifted according to the propogation delay determined from the 
                mean wind speed. If False, all wind direction time series will 
                be in phase.
            ws_mean (float): Mean wind speed. Only used if shift_time is 
                True (m/s).
            seed (int): If defined, used to initialize random number generator.
        """

        # Convert constraint indices to list if passed as int
        if type(inds_constr) == int:
            self.inds_constr = [inds_constr]
        elif inds_constr is not None:
            self.inds_constr = inds_constr

        # Store original coordinates before re-ordering
        if coords_x is not None:
            self.coords_x_orig = coords_x
        if coords_y is not None:
            self.coords_y_orig = coords_y

        if dt is not None:
            self.dt = dt

        # Get number and length of constrained time series
        if wds_constr is not None:
            if len(np.shape(wds_constr)) == 1:
                self.num_constr = 1
                self.t_len = self.dt*len(wds_constr)
                self.wds_constr = wds_constr.reshape(1,len(wds_constr))
            else:
                self.num_constr = np.size(wds_constr,0)
                self.t_len = self.dt*np.size(wds_constr,1)
                self.wds_constr = wds_constr

        # Re-order coordinates so that constraints are first.
        if (inds_constr is not None) | (coords_x is not None) | (coords_y is not None):
            self.inds_new = self.inds_constr+[i for i in range(len(self.coords_x_orig)) if i not in self.inds_constr]
            coords_x = np.array([self.coords_x_orig[i] for i in self.inds_new])
            coords_y = np.array([self.coords_y_orig[i] for i in self.inds_new])

        super().reinitialize_stochastic_wind_directions(
                 coords_x=coords_x,
                 coords_y=coords_y,
                 t_len=self.t_len,
                 dt=dt,
                 wd_std=None,
                 wd_mean=wd_mean,
                 alpha_lat=alpha_lat,
                 alpha_lon=alpha_lon,
                 include_low_freq_wdirs=include_low_freq_wdirs,
                 f_c=f_c,
                 n_ord=n_ord,
                 shift_time=shift_time,
                 ws_mean=ws_mean,
                 seed=seed
                 )

        # Revert total number of samples to length of constrained time series
        self.num_samples_tot = self.num_samples

        # If time shifts are used, determine allowable length of final time series
        if self.shift_time:
            self.num_samples_trunc = self.num_samples - np.max(self.time_delays)

        # Create one-sided frequency vector
        self.freqs = np.arange(1/self.num_samples_tot,0.5+1/self.num_samples_tot,1/self.num_samples_tot)/self.dt

        # Calculate phases and PSDs for constrained time series along with 
        # std. dev. of first time series
        if wds_constr is not None:
            self.freq_mat_constr = np.zeros((self.num_constr,len(self.freqs)),dtype=complex)
            self.psd_mat_constr = np.zeros((self.num_constr,len(self.freqs)),dtype=complex)
            
            # Get std. dev. of first time series for proper scaling
            self.wd_std = np.std(self.wds_constr[0,:])

            for i in range(self.num_constr):
                dft_constr = np.fft.fft(self.wds_constr[i,:])
                self.freq_mat_constr[i,:] = dft_constr[1:len(self.freqs)+1]/np.abs(dft_constr[1:len(self.freqs)+1])
                self.psd_mat_constr[i,:] = np.abs(dft_constr[1:len(self.freqs)+1])**2

    def generate_timeseries(self):
        """
        Generates constrained correlated stochastic high frequency and low 
        frequency wind direction time series at multiple turbine locations 
        based on the Veers method used in TurbSim. One or more wind direction 
        time series are included as constraints at particular locations. The 
        constrained stochastic time series generation method is based on 
        Rinker, J. "PyConTurb: an open-source constrained turbulence generator," 
        TORQUE, 2018. Spatial coherence is defined using an exponential decay 
        coherence model. The low frequency wind direction time series, meant 
        to be used as FLORIS inputs, are formed by applying a the magnitude of 
        a Butterworth LPF transfer function to the high freqiency wind directions. 

        Returns:
            pandas.DataFrame: A dataframe high frequency and low frequency wind 
            direction time series for each turbine location, along with a time column.
        """
        
        # To store Fourier components for each turbine location
        freq_mat = np.zeros((self.num_turbs,len(self.freqs)),dtype=complex)

        # Correlate phases for each frequency component
        for i_f in range(len(self.freqs)):

            # sqrt of coherence matrix
            G = np.sqrt(self._exponential_coherence_combined(self.freqs[i_f]))

            # Cholesky decomposition
            L = np.linalg.cholesky(G)
            
            # Correlate phases
            phase_vec = np.zeros(self.num_turbs,dtype=complex)

            # Determine phases for constrained locations
            phase_vec[:self.num_constr] = np.matmul(
                np.linalg.inv(L[:self.num_constr,:self.num_constr]), self.freq_mat_constr[:,i_f]
                )

            # Assign random phases for remaining locations
            phase_vec[self.num_constr:] = np.exp(
                np.random.uniform(high=2*np.pi,size=self.num_turbs-self.num_constr)*1.0j
                )

            freq_mat[:,i_f] = np.matmul(L,phase_vec)
        
        # create two-sided frequency matrix
        freq_mat_twoside = self._create_two_sided_freqs(freq_mat)

        # Store wind direction time series for each turbine location
        wdirs = np.zeros((self.num_turbs,self.num_samples_tot))
        wdirs_lowfreq = np.zeros((self.num_turbs,self.num_samples_tot))

        # Determine PSDs by interpolating from PSDs of constraint time series
        self._interpolate_psds()

        # Generate time series for each turbine location
        for i in range(self.num_turbs):

            # apply high-frequency PSD magnitudes to frequency components 
            freq_vec_orig = freq_mat_twoside[i,:]
            freq_vec = self._apply_freq_magnitudes_twoside(freq_vec_orig,self.psd_mat[i,:])
            
            # determine scaling factor to get standard deviation of 1
            if i == 0:
                scale_const_total = self.num_samples_tot/np.sqrt(np.sum(np.abs(freq_vec)**2))
            
             # same scaling for all turbine locations
            freq_vec = scale_const_total*freq_vec

            # create time series of high-frequency wind direction
            wdirs[i,:] = (self.wd_std*np.real(np.fft.ifft(freq_vec))+self.wd_mean) % 360

            # apply low-frequency PSD magnitudes to frequency components
            # freq_vec = freq_mat_twoside[i,:]
            freq_vec = self._apply_freq_magnitudes_twoside(
                freq_vec_orig,
                self.psd_mat[i,:]*(self._butterworth_lpf_mag()**2)
                )
            
            # same scaling for all turbine locations
            freq_vec = scale_const_total*freq_vec

            # create time series of low-frequency wind direction
            wdirs_lowfreq[i,:] = (self.wd_std*np.real(np.fft.ifft(freq_vec))+self.wd_mean) % 360

        # Apply time shifts to account for advection time downstream
        if self.shift_time:
            for i in range(self.num_turbs):
                wdirs[i,:self.num_samples_trunc] = (
                    wdirs[i,self.time_delays[i]:self.time_delays[i] + self.num_samples_trunc]
                )
                wdirs_lowfreq[i,:self.num_samples_trunc] = (
                    wdirs_lowfreq[i,self.time_delays[i]:self.time_delays[i] + self.num_samples_trunc]
                )
            wdirs = wdirs[:,:self.num_samples_trunc]
            wdirs_lowfreq = wdirs_lowfreq[:,:self.num_samples_trunc]

        # Save as dataframe
        if self.include_low_freq_wdirs:
            df_hf = pd.DataFrame(wdirs.T,columns=['wd_%03d' % i for i in range(self.num_turbs)])
            df_lf = pd.DataFrame(wdirs_lowfreq.T,columns=['wd_lowfreq_%03d' % i for i in range(self.num_turbs)])
            df = pd.concat([df_hf,df_lf],1)
        else:
            df = pd.DataFrame(wdirs.T,columns=['wd_%03d' % i for i in range(self.num_turbs)])

        # Create time column
        time_col = np.arange(0.,self.t_len,self.dt)
        df['time'] = time_col[:len(df)]

        # sort column names
        df = df.reindex(sorted(df.columns), axis=1)

        return df