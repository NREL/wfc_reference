# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
import turtle

import numpy as np
import pandas as pd

from floris.tools import floris_interface as wfct
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)


def load_floris():
    root_path = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(root_path, "example_floris_input.yaml")
    fi = wfct.FlorisInterface(fn)  # Load FLORIS object
    return fi


if __name__ == "__main__":
    # Set up FLORIS model
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    # Optimimize over a (very) rough grid
    wd = np.arange(0., 360., 30.)
    ws = np.array([6., 9.], dtype=float)
    ti = np.array([0.04, 0.10], dtype=float)
    X, Y, Z = np.meshgrid(wd, ws, ti)
    
    df_opt = pd.DataFrame()
    for turbulence_intensity in ti:
        fi.reinitialize(wind_directions=wd, wind_speeds=ws, 
            turbulence_intensity=turbulence_intensity)
        yaw_opt = YawOptimizationSR(
            fi=fi,
            minimum_yaw_angle=0.,
            maximum_yaw_angle=25.,
            exclude_downstream_turbines=True  # New functionality in PR #245
        )
        df_opt_new = yaw_opt.optimize()
        df_opt = pd.concat((df_opt, df_opt_new))
    print(df_opt)

    # Format dataframe
    yaw_angles = df_opt["yaw_angles_opt"].values
    yaw_angles = np.vstack(yaw_angles)  # Stack to get a [:, num-turbs] matrix
    yaw_angles = np.array(yaw_angles, dtype=float)  # Force 'float' type
    yaw_cols = ['yaw_%03d' % ti for ti in range(num_turbs)]

    # Get a mimimal dataframe
    df_opt_minimal = df_opt\
        [['wind_speed', 'wind_direction', 'turbulence_intensity']]\
        .copy().rename(
            columns={'wind_speed':'ws', 
                     'wind_direction':'wd', 
                     'turbulence_intensity':'ti'}
        )
    df_opt_minimal[yaw_cols] = yaw_angles

    # Save to a .csv file
    root_path = os.path.dirname(os.path.abspath(__file__))
    df_opt_minimal.to_csv(
        os.path.join(root_path, 'df_yaw_lookup_table.csv'), index=False
    )
