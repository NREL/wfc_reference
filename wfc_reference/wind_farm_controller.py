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

from scipy import interpolate

from wfc_reference.helpers import wrap_180

class wind_farm_controller():
    def __init__(
        self,
        df_yaw_lut,  # dataframe with yaw look-up table
        df_upstream,  # dataframe showing which turbines are upstream for what wind directions
        num_turbs,  # Number of turbines
        consensus_dict=None,  # Consensus input parameters
        wakesteering_dict=None, # Wake steering controller parameters
        use_consensus = True, # If True, consensus wind direction estimates will be used. If False, use raw wd values.
        toggle_freq=35,  # Toggle frequency of baseline/wind farm controlled operation
        verbose=False  #
    ):
        if verbose:
            print('Initializing wind farm controller')

        # Save variables to self
        self.num_turbs = num_turbs
        self.df_upstream = df_upstream

        # Initialize yaw offset interpolant
        self._setup_yaw_lut_interpolant(df=df_yaw_lut)

        # Initialize consensus object, if possible
        if consensus_dict is None:
            # If nothing specified, skip consensus
            self.consensus_filtering = None
        else:
            # First define a consensus filtering object which contains
            # the optimization functions that determine an optimal solution
            # in wind directions among the turbines for a given wind direction
            # input array.
            from consensus_filtering import consensus_filtering as cf
            consensus_filtering_object = cf.consensus_filtering(
                layout_x=consensus_dict["layout_x"],
                layout_y=consensus_dict["layout_y"],
                cluster_distance=consensus_dict["cluster_distance"],
                cluster_method=consensus_dict["cluster_method"],
                consensus_method=consensus_dict["consensus_method"],
                tuning_params=consensus_dict["tuning_params"]
            )

            # Now load a wrapper class that basically time averages the
            # incoming wind direction measurements before feeding them
            # in to the consensus algorithm. Also, it can zero-order-hold
            # wind direction consensus estimates.
            from consensus_filtering import consensus_wrapper as cw
            cw_consensus = cw.moving_average_wrapper(
                consensus_filtering_obj=consensus_filtering_object,
                initial_wd_array=consensus_dict["initial_wd_array"],
                moving_avg_time_window=consensus_dict["movingavg_t"],
                consensus_update_rate=consensus_dict["update_rate"]
            )

            # Save to self
            self.consensus_filtering = cw_consensus

        # Assign WFC control mode
        self.use_consensus = use_consensus

        # Higher-level toggle enabling/disabling wind farm control
        self.toggle = False  # Toggle: False: disabled, True: enabled
        self.toggle_counter = 0  # Counter for the toggle
        self.toggle_freq = toggle_freq  # Switch every [x] calls

        if wakesteering_dict is None:
            # disable wake steering
            self.wakesteering_mode = "off"
        else:
            self.wakesteering_mode = wakesteering_dict["wakesteering_mode"]
            
            if wakesteering_dict["wd_filt_tc"] is not None:
                self.use_wd_filt = True
                self.alpha_wd = np.exp(-1*wakesteering_dict["dt"]/wakesteering_dict["wd_filt_tc"])
                self.wd_filt_cos = np.cos(np.radians(wakesteering_dict["wd_initial"]))
                self.wd_filt_sin = np.sin(np.radians(wakesteering_dict["wd_initial"]))
            else:
                self.use_wd_filt = False

            if wakesteering_dict["ws_filt_tc"] is not None:
                self.use_ws_filt = True
                self.alpha_ws = np.exp(-1*wakesteering_dict["dt"]/wakesteering_dict["ws_filt_tc"])
                self.ws_filt = wakesteering_dict["ws_initial"]
            else:
                self.use_ws_filt = False 

            if wakesteering_dict["hyst_size"] is not None:
                self.wd_array_hyst = wakesteering_dict["wd_initial"]

            self.hyst_directions = wakesteering_dict["hyst_directions"] # direction for each turbine where hysteresis is applied
            self.hyst_size = wakesteering_dict["hyst_size"] # yaw offset LUT hysteresis size (in deg)

        # Save other options
        self.verbose = verbose

    def step(self, raw_wd_array, ws_array, ti_array, status_array):
        # Deal with switching between baseline and controlled operation
        self.toggle_counter += 1
        if self.toggle_counter > self.toggle_freq:
            self.toggle_counter = 1  # Reset
            self.toggle = not bool(self.toggle)  # Switch

        # We must call consensus every run to take care of time averaging,
        # even when we don't actually apply these measurements (toggle0=0).
        if self.use_consensus:
            wd_array_consensus, bias_est = self.estimate_wd_consensus(
                raw_wd_array=raw_wd_array,
                status_array=status_array
                )
        else:
            wd_array_consensus = raw_wd_array
            bias_est = np.zeros(self.num_turbs)

        # Similarly, call wake steering controller every run to handle LPF steps
        # Now determine yaw offset setpoints

        if self.wakesteering_mode == "consensus":
            wd_array_wakesteer = wd_array_consensus
        elif self.wakesteering_mode == "raw":
            wd_array_wakesteer = raw_wd_array

        if (self.wakesteering_mode == "consensus") | (self.wakesteering_mode == "raw"):
            yaw_angles_wakesteer = self.get_yaw_offsets_wakesteering(
                wd_array=wd_array_wakesteer,
                ws_array=ws_array,
                ti_array=ti_array,
                status_array=status_array
            )
        else:
            yaw_angles_wakesteer = np.zeros_like(raw_wd_array)

        if self.toggle == 1:
            # Overwrite wind direction measurement with consensus estimates
            wd_array = wd_array_consensus
            yaw_angles = yaw_angles_wakesteer
        else:
            wd_array = raw_wd_array
            yaw_angles = np.zeros_like(raw_wd_array)

        return wd_array, yaw_angles, self.toggle, bias_est

    def estimate_wd_consensus(self, raw_wd_array, status_array):
        if self.verbose:
            print("Calculating consensus wind directions...")

        status_array = np.array(status_array, dtype=bool)
        raw_wd_array = np.array(raw_wd_array, dtype=float)
        raw_wd_array[status_array==False] = np.nan

        if self.consensus_filtering is None:
            wd_consensus_array = raw_wd_array
            bias_est = np.zeros(self.num_turbs)
        else:
            x_est, bias_est = self.consensus_filtering.step(raw_wd_array)
            wd_consensus_array = x_est

        return wd_consensus_array, bias_est

    def _setup_yaw_lut_interpolant(self, df):
        # Save variable to self
        self.df_yaw_lut = df

        # Copy points from 0 deg to 360 deg for linear interpolation
        df_wrap = df[df['wd'] == 0].copy()
        df_wrap['wd'] = 360.
        df = df.append(df_wrap, ignore_index=True)
        df = df.reset_index(drop=True)

        # Create LUT points to interpolate on
        points = df[['wd', 'ws', 'ti']]

        # 'points' looks something like:
        #            wd   ws    ti
        #    0      0.0  6.0  0.03
        #    1      0.0  9.0  0.03
        #    2     10.0  6.0  0.03
        #    3     10.0  9.0  0.03
        #    4     20.0  6.0  0.03
        #    ..     ...  ...   ...
        #    139  330.0  9.0  0.07
        #    140  340.0  6.0  0.07
        #    141  340.0  9.0  0.07
        #    142  350.0  6.0  0.07
        #    143  350.0  9.0  0.07

        # Set up a linear interpolant for the yaw angle of each turbine
        num_turbs = self.num_turbs
        F = [None for _ in range(num_turbs)]
        for ii in range(num_turbs):
            F[ii] = interpolate.LinearNDInterpolator(
                points=points,  # 3D coordinates of LUT
                values=df['yaw_%03d' % ii],  # Output of LUT (yaw angles of turbine ii)
                fill_value=0.  # Set to yaw angle to 0.0 outside of LUT range
            )

        self.yaw_interpolants = F

    def _get_upstream_turbines(self, wd):
        # Return indices of upstream turbines as a function
        # of the ambient wind direction.
        df = self.df_upstream

        """
        Example of df_upstream:
                wd_min  wd_max                       turbines
            0      0.0     1.0      [2, 3, 5, 11, 12, 17, 19]
            1      1.0     1.8  [2, 3, 5, 11, 12, 16, 17, 19]
            2      1.8     6.3     [2, 5, 11, 12, 16, 17, 19]
            3      6.3    11.9         [2, 5, 11, 12, 17, 19]
            4     11.9    25.4     [2, 5, 11, 12, 17, 18, 19]
            ..     ...     ...                            ...
            70   329.0   329.3      [1, 2, 5, 11, 12, 17, 19]
            71   329.3   341.9         [2, 5, 11, 12, 17, 19]
            72   341.9   357.4     [2, 5, 11, 12, 13, 17, 19]
            73   357.4   358.9         [2, 5, 11, 12, 17, 19]
            74   358.9   360.0      [2, 3, 5, 11, 12, 17, 19]
        """

        # Find the right row and extract turbines
        if wd > 0.:
            cond = (wd > df['wd_min']) & (wd <= df['wd_max'])
        else:
            cond = (wd >= df['wd_min']) & (wd <= df['wd_max'])
        turbines = df.loc[cond, 'turbines'].values[0]
        return turbines  # a list of turbine indices (integers)

    def interp_optimal_yaw_angles(
        self, wd_array, ws_array, ti_array, status_array
    ):
        if self.verbose:
            print("Calculating optimal yaw angles...")

        num_turbs = self.num_turbs

        # Format inputs to numpy arrays
        wd_array = np.array(wd_array, dtype=float)
        ws_array = np.array(ws_array, dtype=float)
        ti_array = np.array(ti_array, dtype=float)
        status_array = np.array(status_array, dtype=int)

        # Get list of valid turbines vased on status
        valid_inds = list(np.where(status_array != 0)[0])

        # Define ambient wind speed and turbulence intensity
        wd_mean = np.nanmean(wd_array[valid_inds])
        upstream_turbines = self._get_upstream_turbines(wd_mean)
        ws_ambient = np.nanmean(ws_array[[i for i in upstream_turbines if i in valid_inds]])
        ti_ambient = np.nanmean(ti_array[[i for i in upstream_turbines if i in valid_inds]])

        # Interpolate optimal yaw angle for each turbine
        yaw_angles = np.zeros(self.num_turbs)
        for ti in range(self.num_turbs):
            yaw_interpolant = self.yaw_interpolants[ti]
            yaw_angles[ti] = yaw_interpolant(
                wd_array[ti], ws_ambient, ti_ambient
            )

        # ...
        # Currently, the above code is a simple placeholder for direct
        # yaw angle interpolation. However, in practice, we may want to
        # accommodate for turbines that are offline (status==0). Ideally,
        # for every combination of online/offline turbines, we will have
        # a different optimal set of yaw angles (different optimal look-up
        # table). Of course, due to the curse of dimensionality, we may
        # simply assume we stop all wake steering if three or more turbines
        # are offline.
        # ...

        return yaw_angles

    def get_yaw_offsets_wakesteering(
        self, wd_array, ws_array, ti_array, status_array
    ):
        if self.verbose:
            print("Running wake steering controller...")

        # Update wind direction and wind speed filters
        if self.use_wd_filt:
            self.wd_filt_cos = (1 - self.alpha_wd) * np.cos(np.radians(wd_array)) + self.alpha_wd * self.wd_filt_cos
            self.wd_filt_sin = (1 - self.alpha_wd) * np.sin(np.radians(wd_array)) + self.alpha_wd * self.wd_filt_sin
            wd_array = np.degrees(np.arctan2(self.wd_filt_sin, self.wd_filt_cos)) % 360

        if self.use_ws_filt:
            self.ws_filt = (1 - self.alpha_ws) * ws_array + self.alpha_ws * self.ws_filt
            ws_array = self.ws_filt

        # hysteresis logic
        if self.hyst_size is not None:
            # TODO: consider a more efficient way of updating hysteresis directions 
            for ti in range(self.num_turbs):
                if all([(np.abs(wrap_180(wd_array[ti]-hyst_dir)) > self.hyst_size) for hyst_dir in self.hyst_directions[ti]]):
                    self.wd_array_hyst[ti] = wd_array[ti]
        else:
            self.wd_array_hyst = wd_array

        yaw_angles = self.interp_optimal_yaw_angles(
            wd_array=self.wd_array_hyst,
            ws_array=ws_array,
            ti_array=ti_array,
            status_array=status_array
        )

        return yaw_angles