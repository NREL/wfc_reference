# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from datetime import timedelta as td
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from floris.tools import floris_interface as wfct
from floris.utilities import wrap_360

from wfc_reference.wind_farm_controller import wind_farm_controller
from wfc_reference.wind_turbine_controller import wind_turbine_controller
from wfc_reference.visualization import visualize_flow_field
from wfc_reference.stochastic_wind_field import stochastic_wind_directions, constrained_stochastic_wind_directions


def load_data():
    root_path = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(root_path, "example_dataset.ftr")
    df = pd.read_feather(fn)  # Load feather file
    df = df.head(1800)  # Reduce to first 2 hours
    df = df.dropna(how="any")  # Drop NaN entries
    df = df.reset_index(drop=True)  # Reset indices

    # set mean wind direction to 160 deg
    df['wind_direction'] = (df['wind_direction'] - np.mean(df['wind_direction']) + 160.0) % 360.0

    # set mean wind speed to 8 m/s
    df['wind_speed'] = df['wind_speed'] - np.mean(df['wind_speed']) + 8.0

    # Copy conditions over to each turbine
    num_turbs = 7
    for ti in range(num_turbs):
        df['wd_%03d' % ti] = df['wind_direction']
        df['ws_%03d' % ti] = df['wind_speed']
        df['ti_%03d' % ti] = 0.06

    # Disturb particular turbines
    df['wd_003'] = wrap_360(df['wd_003'] + 4.81)
    df['wd_005'] = wrap_360(df['wd_005'] - 8.98)

    # Add status flags and mimic downtime
    for ti in range(num_turbs):
        df['status_%03d' % ti] = int(1)
        # remove downtime for now
        if False:
            no_downtimes = np.random.randint(0, 5)
            ids_down = []
            for _ in range(no_downtimes):
                dt_lb = np.random.randint(0, df.shape[0] - 1)
                dt_length = np.random.randint(int(3 * 60), int(45 * 24 * 60))
                dt_ub = np.min([dt_lb + dt_length, df.shape[0] - 1])
                ids_down.extend(np.arange(dt_lb, dt_ub, 1, dtype=int))
            df.loc[ids_down, 'status_%03d' % ti] = int(0)

    return df

def generate_stochastic_wds(df, fi, use_constraints=False):
    # Generates stochastic wind directions for each turbine location and overwrites 
    # the original wind directions stored in df. If use_constraints is True, the first 
    # turbine's wind directions will be constrained to be the measured wind direction 
    # time series in df. 

    # Get turbine coordinates
    coords_x, coords_y = fi.get_turbine_layout()

    # Time variables
    t_len = len(df) # length in seconds
    dt = 1. # time step

    # Define wind field parameters
    wd_mean = df['wind_direction'].mean() # mean wind direction
    wd_std = 11. # wind direction std. dev. 
    ws_mean = df['wind_speed'].mean() # mean wind speed

    # Shift time series to emulate propogation time of wind field downstream
    shift_time = False

    # Set up stochastic wind direction generator
    if use_constraints:
        wd_gen = constrained_stochastic_wind_directions(
            coords_x=coords_x,
            coords_y=coords_y,
            wds_constr=df.wind_direction.values,
            inds_constr=0,
            dt=dt,
            wd_mean=wd_mean,
            alpha_lat=0.2,
            alpha_lon=0.1,
            shift_time=shift_time,
            ws_mean=ws_mean
        )
    else:
        wd_gen = stochastic_wind_directions(
            coords_x=coords_x,
            coords_y=coords_y,
            t_len=t_len,
            dt=dt,
            wd_std=wd_std,
            wd_mean=wd_mean,
            shift_time=shift_time,
            ws_mean=ws_mean
        )

    df_wd = wd_gen.generate_timeseries()

    # Overwrite wind direction time series for each turbine location
    for ti in range(num_turbs):
        df['wd_%03d' % ti] = df_wd['wd_%03d' % ti]

    # Disturb particular turbines
    df['wd_003'] = wrap_360(df['wd_003'] + 4.81)
    df['wd_005'] = wrap_360(df['wd_005'] - 8.98)

    return df

def load_floris():
    root_path = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(root_path, "example_floris_input.yaml")
    fi = fi = wfct.FlorisInterface(fn)  # Load FLORIS object
    return fi


if __name__ == "__main__":
    # Evaluation settings
    plot_flowfields = False
    # Set use_stochastic_wind_dirs to True to generate stochastic wind direction time 
    # series. If so, set use_stochastic_constraints to True to constrain the wind 
    # directions for the first turbine to be the original measured wind direction time series.
    
    dt = 1.0 # time step
    use_stochastic_wind_dirs = True
    use_stochastic_constraints = True
    root_path = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(root_path, "figures")
    os.makedirs(fig_path, exist_ok=True)

    # Load data
    df = load_data()
    N = df.shape[0]

    # Set up FLORIS model
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    # Generate correlated stochastic wind direction time series at each turbine location
    if use_stochastic_wind_dirs:
        df = generate_stochastic_wds(df, fi, use_constraints=use_stochastic_constraints)

    # Load yaw LUT
    df_yaw_lut = pd.read_csv(
        os.path.join(root_path, "df_yaw_lookup_table.csv")
        )

    # Get upstream turbines from precalculated table
    df_upstream = pd.read_csv(
        os.path.join(root_path, "df_upstream.csv")
    )
    df_upstream['turbines'] = [eval(x) for x in df_upstream['turbines']] # convert string to list

    wd_initial = np.array(df.loc[0, ['wd_%03d' % ti for ti in range(num_turbs)]]).astype(float)
    consensus_dict = {
        "layout_x": fi.layout_x,
        "layout_y": fi.layout_y,
        "cluster_distance": [3],
        "cluster_method": "nearest",
        "consensus_method": "weighted_cluster_average",
        "tuning_params": dict(
            {
                "rho": 1.0,
                "lam": 1.0,
                "lam1": 1.0,
                "eps": 0.0,
                "maxiter": 30
            }
        ),
        "initial_wd_array": wd_initial,  # Initial estimate/value for the consensus-based wind directions
        "movingavg_t": 20,  # Number of measurements to average over (measurements from past [x] function calls)
        "update_rate": 5,  # How often to update consensus estimates (every [x] function calls)
    }
    consensus_dict = None # REMOVE to use consensus

    ws_initial = np.array(df.loc[0, ['ws_%03d' % ti for ti in range(num_turbs)]]).astype(float)
    wakesteering_dict = {
        "wd_filt_tc": 30.0,  # time constant for filtering wind direction input to yaw offset LUT. If None, filtering will not be used.
        "wd_initial": wd_initial,  # List or array of wind directions used to initialize wake steering filters and hysteresis logic, if they are used
        "ws_filt_tc": 60.0,  # time constant for filtering wind speed input to yaw offset LUT. If None, filtering will not be used.
        "ws_initial": ws_initial,  # List or array of wind speeds used to initialize filter if ws_filt_tc is not None
        "dt": dt,  # Simulation time step (s)
        "hyst_size": None,  # yaw offset LUT hysteresis size (in deg). If None, hysteresis will not be used.
        "hyst_directions": None,  # dictionary containing crossover directions for each turbine where hysteresis should be applied
        "wakesteering_mode": "raw"  # If "consensus", use consensus wind directions as input to wake steering  controller. If "raw", use raw wind directions as input. If "off", do not use wake steering.
    }

    use_consensus_for_yaw_control = True # If True, consensus wind direction will be used for yaw control. Otherwise they may still be used as inputs to the wake steering controller.
    
    # Set up wind farm controller
    wfc = wind_farm_controller(
        df_yaw_lut=df_yaw_lut,
        df_upstream=df_upstream,
        num_turbs=num_turbs,
        consensus_dict  = consensus_dict,
        wakesteering_dict = wakesteering_dict,
        use_consensus   = True,
        toggle_freq=N,  # Set so there's only a single toggle per simulation
    )

    # Set the toggle to True or False
    wfc.toggle = True

    # Set up wind turbine controllers
    wtcs = [None for _ in range(num_turbs)]
    for ti in range(num_turbs):
        # Baseline yaw controller settings
        yaw_controller_dict = ({
            "yaw_init": df.iloc[0]["wd_%03d" % ti],
            "wd_init": df.iloc[0]["wd_%03d" % ti],
            "time_const": 30.,  # in [s]
            "deadband": 8.,
            "yaw_rate": 0.3,  # deg/s
            "dt": dt
            })

        # Yaw controller settings with consensus enabled
        yaw_controller_opt = ({
            "time_const": 30.,
            "deadband": 8.,
            "yaw_rate": 0.3,
            "dt": dt
            })

        wtcs[ti] = wind_turbine_controller(
            turbine_name="WTG_%03d" % ti,
            yaw_dict_default=yaw_controller_dict,
            yaw_dict_opt=yaw_controller_opt,
            sampling_time=td(seconds=dt)
            )

    # Do timeseries simulation
    df_timeseries = pd.DataFrame({})
    for ii in range(N):
        # ARTIFICIAL: Print a progress update every 50 steps
        if np.remainder(ii, 50) == 0:
            print("Evaluated %d/%d time steps." % (ii, N))

        # ARTIFICIAL:
        # Get ambient conditions and turbine performance from artificial
        # dataset. Then, copy those conditions over to the individual wind
        # turbines.
        current_time = df.loc[ii, "time"]
        wd_array = np.array(df.loc[ii, [c for c in df.columns if "wd_" in c]])
        ws_array = np.array(df.loc[ii, [c for c in df.columns if "ws_" in c]])
        ti_array = np.array(df.loc[ii, [c for c in df.columns if "ti_" in c]])
        status_array = np.array(df.loc[ii, [c for c in df.columns if "status_" in c]])
        for ti in range(num_turbs):
            vane_angle = wd_array[ti] - wtcs[ti].nacelle_heading
            wtcs[ti]._set_conditions_artificial(
                vane_angle=vane_angle,
                wind_speed=ws_array[ti],
                turbulence_intensity=ti_array[ti],
                status=status_array[ti]
                )

        # TURB. CONTROLLER
        # Read the raw turbine sensors and get back a set of variables.
        # In this python example, this is packaged into a dict(), basically
        # containing a set of keys (variable names) and values (variable
        # values).
        turb_measurements = [{} for _ in range(num_turbs)]
        for ti in range(num_turbs):
            turb_measurements[ti] = wtcs[ti].read_turbine_sensors()

        # FARM CONTROLLER
        # Gather required turbine measurements and pass them to the wind farm
        # controller as arrays of length num_turbs. Then, internally, get
        # consensus WDs and optimal yaw angles if toggle=True, or set wd_array,
        # set wd_array = wd_a unchanged and yaw_angles=0 if toggle=False.
        nac_heading = [tm["nacelle_heading"] for tm in turb_measurements]
        vane_angle = [tm["nacelle_vane_angle"] for tm in turb_measurements]
        wd_raw_array = np.array(nac_heading) + np.array(vane_angle)
        ws_array = np.array([tm["wind_speed"] for tm in turb_measurements])
        ti_array = np.array([tm["turbulence_intensity"] for tm in turb_measurements])
        status_array = [tm["status"] for tm in turb_measurements]

        wd_array, yaw_angles, wfc_toggle, bias_est = wfc.step(
            wd_raw_array, ws_array, ti_array, status_array
            )

        # TURB. CONTROLLER
        # Overwrite local turbine vane measurements using the consensus-based
        # wind direction estimates. This should more accurately represent the
        # true wind condition in the farm. Also, if the wind farm controller
        # is toggled on, then we use stricter yaw parameters for improved
        # tracking, and pass optimal yaw offsets as target_vane_offset.
        for ti in range(num_turbs):
            
            # Use consensus estimated wd to get a better vane measurement if use_consensus_for_yaw_control is True
            if use_consensus_for_yaw_control:
                corrected_vane = wd_array[ti] - wtcs[ti].nacelle_heading
            else:
                corrected_vane = None

            # Step forward in time for the turbine controller
            if wfc_toggle:
                wtcs[ti].step(
                    corrected_vane_angle=corrected_vane,
                    target_vane_offset=yaw_angles[ti],
                    wfc_toggle=wfc_toggle
                    )
            else:
                wtcs[ti].step(
                    corrected_vane_angle=None,
                    target_vane_offset=0.0,
                    wfc_toggle=wfc_toggle
                    )

        # ARTIFICIAL: Save timeseries
        dict_merged = {}  # Artificial: empty dict for saving timeseries
        for ti in range(num_turbs):
            # # Overwrite turbine power measurements with FLORIS predictions
            # gen_pow_array[ti] = pow_array[ti]

            # Append turbine measurements to dict_merged
            ti_dict = {k + "_%03d" % ti: v for k, v in turb_measurements[ti].items()}
            dict_merged.update(ti_dict)

            # Add wind farm controller outputs to dict_merged
            dict_merged["wd_consensus_%03d" % ti] = wd_array[ti]
            dict_merged["yaw_setpoint_%03d" % ti] = yaw_angles[ti]
            dict_merged["wakesteering_lut_wd_input_%03d" % ti] = wfc.wd_array_hyst[ti]
        
        df_timeseries = pd.concat((df_timeseries, 
            pd.DataFrame({k: [v] for k, v in dict_merged.items()})), 
            ignore_index=True)

        # ARTIFICIAL: Calculate and save flow fields using FLORIS
        if plot_flowfields:
            fig, ax = visualize_flow_field(
                fi=fi,
                yaw_angles=[wtcs[ti].vane_angle for ti in range(num_turbs)],
                wd=np.mean(wd_array),
                ws=np.mean(ws_array),
                ti=np.min(ti_array),
                title="Time %s" % str(current_time))
            plt.savefig(os.path.join(fig_path, "out_%03d.png" % ii))
            plt.close(fig)

    # ARTIFICIAL: Plot wind direction and yaw trajectories
    for ti in range(num_turbs):
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(df["time"], df["wd_%03d" % ti], "-o", label="Raw turbine wd (deg)")
        ax[0].plot(df["time"], df_timeseries["wd_consensus_%03d" % ti], "-o", label="Consensus wd (deg)")
        ax[0].plot(df["time"], df_timeseries["wd_consensus_%03d" % ti] - df_timeseries["yaw_setpoint_%03d" % ti], "-o", label="Assigned wd (deg)")
        ax[0].plot(df["time"], df_timeseries["nacelle_heading_%03d" % ti], "-o", label="Nacelle heading (deg)")
        # ax[0].set_title("Turbine %s" % str(turb_names[ti]))
        ax[0].plot(df["time"], df_timeseries["wakesteering_lut_wd_input_%03d" % ti], label="Input wd to lookup table (deg)")
        ax[0].legend()
        ax[0].set_title(f"Turbine {ti}")

        ax[1].plot(df["time"], df_timeseries["yaw_setpoint_%03d" % ti], "-o", label="Nac. vane setpoints (deg)")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Nacelle vane setpoints (deg)")
        # plt.savefig(os.path.join(fig_path, "turbine_%03d" % ti))
        # plt.close(fig)
        plt.show()
