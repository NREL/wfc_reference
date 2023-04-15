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


class hysteresis_filter():
    def __init__(self, hyst_zones, initial_wd_array):
        # Initialize hysteresis filter
        self.hyst_zones = hyst_zones
        self.nturbs = len(initial_wd_array)

        # Check if initial conditions fall in any hysteresis zone
        for ti in range(self.nturbs):
            for h in self.hyst_zones[ti]:
                if ((initial_wd_array[ti] > h[0]) & (initial_wd_array[ti] < h[1])):
                    print("Initial condition for turbine {:d} falls in a hysteresis region.".format(ti))
                    if (initial_wd_array[ti] - h[0]) < (h[1] - initial_wd_array[ti]):
                        initial_wd_array[ti] = h[0]
                    else:
                        initial_wd_array[ti] = h[1]
                    print("Assumed nearest value of {:.1f}".format(initial_wd_array[ti]))

        self.wd_array_saturated = initial_wd_array
        self.nturbs = len(initial_wd_array)

    def filter(self, wd_array_in):

        wd_array_out = np.array(wd_array_in, dtype=float, copy=True)  # By default, return input values assume no hysteresis
        for ti in range(self.nturbs):
            turbine_is_saturated = False  # Initialize as False
            for h in self.hyst_zones[ti]:
                if (wd_array_in[ti] > h[0]) & (wd_array_in[ti] < h[1]):
                    turbine_is_saturated = True  # Falls within hysteresis region
                    if self.wd_array_saturated[ti] <= h[0]:
                        wd_array_out[ti] = h[0]  # If previously on left side of hysteresis, return left bound value
                    elif self.wd_array_saturated[ti] >= h[1]:
                        wd_array_out[ti] = h[1]  # If previously on right side of hyst sector, return right bound value
                    else:
                        raise UserWarning("Incompatible hysteresis zone/wind direction. This should never happen.")

            # If this turbine does not fall in any hysteresis zones, the saturated value is the same as the unsaturated value
            if not turbine_is_saturated:
                self.wd_array_saturated[ti] = wd_array_in[ti]
        
        return wd_array_out
