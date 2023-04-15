# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
from time import perf_counter as timerpc

import matplotlib.pyplot as plt
import numpy as np

import floris.tools as wfct
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


def load_floris():
    # Instantiate the FLORIS object
    root_path = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.floris_interface.FlorisInterface(
        os.path.join(root_path, "example_input.json")
    )

    # Set turbine locations to 3 turbines in a row
    D = fi.floris.farm.turbines[0].rotor_diameter
    layout_x = [0, 7 * D, 14 * D, 0, 7 * D, 14 * D]
    layout_y = [0, 0, 0, 5 * D, 5 * D, 5 * D]
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    return fi


def _optimize_single_case(fi, x0, bnds, Ny_passes=[5, 5]):
    # SR parameters
    opt_options = {
        "Ny_passes": Ny_passes,
        "refine_solution": False,
        "refine_method": "SLSQP",
        "refine_options": {
            "maxiter": 10,
            "disp": False,
            "iprint": 1,
            "ftol": 1e-7,
            "eps": 0.01,
        },
    }

    # Load optimizer
    yaw_opt = YawOptimizationSR(
        fi=fi,
        yaw_angles_baseline=x0,
        bnds=bnds,
        opt_options=opt_options,
        include_unc=False,  # No wind direction variability in floris simulations
        exclude_downstream_turbines=True,  # Exclude downstream turbines automatically
        cluster_turbines=True,  # Do not bother with clustering
    )

    # Run optimization
    yaw_angles_opt = yaw_opt.optimize()

    return yaw_angles_opt


def optimize_windrose_slope_constrained(fi, wd_array, yaw_lb, yaw_ub, maximum_slope, Ny_passes_init=[11, 11], Ny_passes_refine=[3, 3]):
    # Start timer
    start_time = timerpc()

    # FLORIS settings
    nturbs = len(fi.layout_x)

    # Initialize bnds
    bnds = [(yaw_lb, yaw_ub) for _ in range(nturbs)]  # Start for entire range

    # Run through all wind directions
    x = np.zeros(nturbs)
    Ny_passes = Ny_passes_init  # Start with high precision
    yaw_angles_opt = np.zeros((len(wd_array), nturbs))
    for ii, wd in enumerate(wd_array):
        # Optimize for current atmospheric conditions and bounds
        fi.reinitialize_flow_field(wind_direction=wd)
        x = _optimize_single_case(fi, x, bnds, Ny_passes=Ny_passes)
        yaw_angles_opt[ii, :] = x  # Save to array

        # Recalculate bounds
        bnds = [(xi - maximum_slope, xi + maximum_slope) for xi in x]
        bnds = [(np.max([b[0], yaw_lb]), np.min([b[1], yaw_ub])) for b in bnds]

        # After first evaluation, reduce search precision
        if ii == 0:
            Ny_passes = Ny_passes_refine  # Then continue with lower precision search range

    print("Optimized yaw angles for {:d} wind directions in {:.2f} seconds.".format(len(wd_array), timerpc() - start_time))
    return yaw_angles_opt


if __name__ == "__main__":
    # Load the FLORIS model
    fi = load_floris()
    nturbs = len(fi.layout_x)

    # Specify maximum slope
    

    # Optimization settings
    wd_array = np.arange(164.0, 210.0, 2.0)
    yaw_lb = 0.0
    yaw_ub = 20.0
    maximum_slope = 4.0  # Maximum step up or down in degrees
    Ny_passes_init = [11, 11]
    Ny_passes_refine = [5, 5]

    # Specify atmospheric conditions
    
    yaw_angles_opt_all = optimize_windrose_slope_constrained(
        fi, wd_array, yaw_lb, yaw_ub, 40.0, Ny_passes_init=[11, 11], Ny_passes_refine=[11, 11])
    yaw_angles_opt_lr = optimize_windrose_slope_constrained(
        fi, wd_array, yaw_lb, yaw_ub, maximum_slope, Ny_passes_init, Ny_passes_refine)
    yaw_angles_opt_rl = optimize_windrose_slope_constrained(
        fi, wd_array[::-1], yaw_lb, yaw_ub, maximum_slope, Ny_passes_init, Ny_passes_refine)[::-1]

    # Now do the dumb method for left to right
    yaw_angles_dumb_lr = [yaw_angles_opt_all[0]]
    for yaw in yaw_angles_opt_all[1::]:
        delta_yaw = yaw - yaw_angles_dumb_lr[-1]
        delta_yaw[delta_yaw > maximum_slope] = maximum_slope  # Constrain to limit
        delta_yaw[delta_yaw < -maximum_slope] = -maximum_slope  # Constrain to limit
        yaw_angles_dumb_lr.append(yaw_angles_dumb_lr[-1] + delta_yaw)
    yaw_angles_dumb_lr = np.vstack(yaw_angles_dumb_lr)

    # Now do the dumb method for right to left
    yaw_angles_dumb_rl = [yaw_angles_opt_all[::-1][0]]
    for yaw in yaw_angles_opt_all[::-1][1::]:
        delta_yaw = yaw - yaw_angles_dumb_rl[-1]
        delta_yaw[delta_yaw > maximum_slope] = maximum_slope  # Constrain to limit
        delta_yaw[delta_yaw < -maximum_slope] = -maximum_slope  # Constrain to limit
        yaw_angles_dumb_rl.append(yaw_angles_dumb_rl[-1] + delta_yaw)
    yaw_angles_dumb_rl = np.vstack(yaw_angles_dumb_rl[::-1])

    # Plot results
    for ti in range(1):
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(3, 5))
        ax[0].plot(wd_array, wd_array * 0.0, "--", color="gray")
        ax[0].plot(wd_array,yaw_angles_opt_all[:, ti], "-o", markersize=3, color="black", label="Optimization unconstrained")
        ax[0].plot(wd_array, yaw_angles_opt_lr[:, ti], label="Optimization constrained, from left to right")
        ax[0].plot(wd_array, yaw_angles_opt_rl[:, ti], label="Optimization constrained, from right to left")

        ax[1].plot(wd_array, wd_array * 0.0, "--", color="gray")
        ax[1].plot(wd_array, yaw_angles_opt_lr[:, ti], label="Optimization constrained, from left to right")
        ax[1].plot(wd_array, yaw_angles_dumb_lr[:, ti], "--o", markersize=2, label="Simple filter over the unconstrained solution, from left to right")
        
        ax[2].plot(wd_array, wd_array * 0.0, "--", color="gray")
        ax[2].plot(wd_array, yaw_angles_opt_rl[:, ti], label="Optimization constrained, from right to left")
        ax[2].plot(wd_array, yaw_angles_dumb_rl[:, ti], "--o", markersize=2, label="Simple filter over the unconstrained solution, from right to left")
        ax[2].set_xlabel("Wind direction (deg)")

        for axi in ax:
            axi.set_ylabel("Yaw offset (deg)")
            axi.grid(True)
            axi.legend(loc="right")
        # ax[0].set_title("Turbine {:02d}".format(ti))

    plt.tight_layout()
    plt.show()