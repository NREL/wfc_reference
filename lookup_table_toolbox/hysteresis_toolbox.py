# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from matplotlib import pyplot as plt
import numpy as np


class hysteresis_toolbox():
    # Initialize class
    def __init__(self, wd_array, yaw_angles, verbose=True, debug=False):
        # Save information about yaw curve to self
        self.wd_array = wd_array
        self.yaw_angles = yaw_angles

        # Other toolbox settings
        self.debug = debug
        self.verbose = verbose

        # Derive additional information
        self.yaw_lb = np.min(yaw_angles)
        self.yaw_ub = np.max(yaw_angles)

        # Initialize empty arrays
        self.hysteresis_wds = []

    def identify_hysteresis_zones(self, min_region_width=2.0, yaw_jump_threshold=9.99) -> list:
        # Extract variables from self
        wd_array = self.wd_array
        yaw_angles = self.yaw_angles
        verbose = self.verbose

        # Define function that identifies hysteresis zones
        switching_points = np.where(np.diff(yaw_angles) > yaw_jump_threshold)[0]
        wd_switching_points = np.mean(wd_array[np.vstack([switching_points, switching_points + 1]).T], axis=1)
        if verbose:
            print("Identified points where optimal offset increases by more than {:.2f} deg over one wind direction step.".format(yaw_jump_threshold))
            print("Center points for hysteresis: {}".format(wd_switching_points))

        # find left limit at each switching point where optimal positive angle hits 0 deg
        hysteresis_wds = []
        for wd_switch_point in wd_switching_points:
            lb = np.max(wd_array[(wd_array < wd_switch_point) & (yaw_angles < 0.1)])
            ub = np.min(wd_array[(wd_array > wd_switch_point) & (yaw_angles > -0.1)])
            if (ub - lb) < min_region_width:
                center_point = 0.50 * lb + 0.50 * ub
                lb = center_point - 0.50 * min_region_width
                ub = center_point + 0.50 * min_region_width
            hysteresis_wds.append((lb, ub))
        
        self.hysteresis_wds = hysteresis_wds
        if verbose:
            print("Identified hysteresis regions: {}".format(hysteresis_wds))

        return hysteresis_wds

    def get_max_yaw_rate(self):
        wd_array = self.wd_array
        yaw_angles = self.yaw_angles
        hysteresis_wds = self.hysteresis_wds
        verbose = self.verbose

        # Check if any hysteresis regions found
        if len(hysteresis_wds) < 1:
            print("INFO: No hysteresis regions found (or'identify_hysteresis_zones' has not yet been run.)")
            print("INFO: The maximum yaw rate can therefore be very high.")

        # Calculate highest sensitivity of the yaw angle vs. wind direction outside of hysteresis regions
        wd_lb = 0
        max_yawrate = 0.0
        for ii in range(len(hysteresis_wds) + 1):
            # Set lower bound
            if ii == 0:
                wd_lb = 0.0
            else:
                wd_lb = hysteresis_wds[ii - 1][1]

            # Set upper bound
            if ii == len(hysteresis_wds):
                wd_ub = 360.0
            else:
                wd_ub = hysteresis_wds[ii][0]  # Left hysteresis bound

            # Calculate maximum yaw rate
            wd_subset = wd_array[(wd_array >= wd_lb) & (wd_array <= wd_ub)]
            yaw_angles_subset = yaw_angles[(wd_array >= wd_lb) & (wd_array <= wd_ub)]
            yaw_rates = np.abs(np.diff(yaw_angles_subset) / np.diff(wd_subset))
            if np.max(yaw_rates) > max_yawrate:
                max_yawrate = np.max(yaw_rates)
                max_yawrate_wd = wd_subset[[np.argmax(yaw_rates), np.argmax(yaw_rates)+1]]

        if verbose:
            print("Maximum yaw offset gradient of {:.1f} in wd sector ({:.1f}, {:.1f})".format(max_yawrate, *max_yawrate_wd))

        return max_yawrate, max_yawrate_wd


    def plot(self, ax=None):
        # Extract variables from self
        wd_array = self.wd_array
        yaw_angles = self.yaw_angles
        hysteresis_wds = self.hysteresis_wds
        yaw_lb = self.yaw_lb
        yaw_ub = self.yaw_ub

        # Plot this turbine's yaw curves
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        ax.plot(wd_array, yaw_angles * 0.0, "--", color='gray')  # Zero line
        ax.plot(wd_array, yaw_angles, '-o', color='black', markersize=2, label="Optimal yaw curve")

        for hi, hyst in enumerate(hysteresis_wds):
            # Only label first hysteresis region
            if hi == 0:
                label="Hysteresis regions"
            else:
                label=None

            ax.fill_betweenx(
                [yaw_lb - 5.0, yaw_ub + 5.0],
                [hyst[0], hyst[0]],
                [hyst[1], hyst[1]],
                color="tab:red",
                alpha=0.3,
                label=label,
            )

        ax.grid(True)
        ax.set_xlabel("Wind direction (deg)")
        ax.set_ylabel("Yaw offset angle (deg)")
        b = (0.5, 1.15)
        ax.set_xlim([wd_array[0], wd_array[-1]])
        ax.set_ylim([yaw_lb - 5.0, yaw_ub + 5.0])
        ax.legend(loc='upper center', bbox_to_anchor=b, ncol=2, fancybox=False, framealpha=0)

        plt.tight_layout()
        return fig, ax
