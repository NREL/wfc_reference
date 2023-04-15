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

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from lookup_table_toolbox.hysteresis_toolbox import hysteresis_toolbox


def extract_yaw_angles_from_csv(df: pd.DataFrame) -> np.ndarray:
    yaw_angles = np.vstack(df["yaw_angles"])
    yaw_angles = [[y.replace("[", "").replace("]", "") for y in yaw_angle] for yaw_angle in yaw_angles]
    yaw_angles = [yaw_angle[0].split(" ") for yaw_angle in yaw_angles]
    yaw_angles = [[float(y) for y in yaw_angle if ((len(y) > 0) & (not y == " "))] for yaw_angle in yaw_angles]
    yaw_angles = np.array(yaw_angles, dtype=float)
    return yaw_angles


if __name__ == "__main__":
    # Load optimized yaw angles for a single turbine
    root_path = os.path.dirname(os.path.abspath(__file__))
    examples_path = os.path.join(root_path, "..", "examples")
    df_opt = pd.read_csv(os.path.join(examples_path, "df_opt_negpos.csv"))  # Optimal yaw angles when bounded to (-20.0, +20.0)
    # df_opt = pd.read_csv(os.path.join(examples_path, "df_opt_neg.csv"))  # Optimal yaw angles when bounded to (-20.0, 0.0)
    # df_opt = pd.read_csv(os.path.join(examples_path, "df_opt_pos.csv"))  # Optimal yaw angles when bounded to (0.0, +20.0)

    ti = 0
    wd_array = np.array(df_opt["wd"].unique(), dtype=float)
    yaw_angles_opt = extract_yaw_angles_from_csv(df_opt)[:, ti]

    # Initialize class
    s = hysteresis_toolbox(wd_array, yaw_angles_opt)

    # Now calculate hysteresis regions
    s.get_max_yaw_rate()  # Calculate max yaw rate before hysteresis
    print(" ")

    s.identify_hysteresis_zones(min_region_width=5.0)
    s.get_max_yaw_rate()  # Maximum yaw rate with hysteresis
    s.plot()
    plt.show()
