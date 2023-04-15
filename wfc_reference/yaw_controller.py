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

# TODO These functions exist already in floris/flasc
def wrap_180(x):
    x = np.where(x <= -180.0, x + 360.0, x)
    x = np.where(x > 180.0, x - 360.0, x)
    return float(x)


def wrap_360(x):
    x = np.where(x < 0.0, x + 360.0, x)
    x = np.where(x >= 360.0, x - 360.0, x)
    return float(x)


class yaw_controller(object):
    """Minimalistic yaw controller example"""
    def __init__(
        self,
        options_dict=({
            'yaw_init': 0.0,
            'wd_init': 0.0,
            'time_const': 30.,
            'deadband': 8.,
            'yaw_rate': 0.3,
            'dt':1.0,
            })
    ):
        # Initialize filtered error and state machine
        self.wd_cos_filt = np.cos(np.deg2rad(options_dict['wd_init']))
        self.wd_sin_filt = np.sin(np.deg2rad(options_dict['wd_init']))
        self.yaw = options_dict['yaw_init']
        self.yaw_state = 0

        self.time_const = options_dict['time_const']  # Set the filter time constant
        self.deadband = options_dict['deadband']  # Set a deadband
        self.yaw_rate = options_dict['yaw_rate']  # Set the yaw rate
        self.dt = options_dict['dt'] # Sampling time for controller

        self.set_lpf()

    def set_lpf(self):

        # Set up low-pass filter for wind direction
        fc = 1 / (2 * np.pi * self.time_const)
        self.lpf_alpha = np.exp(-2 * np.pi * self.dt * fc)

    def compute(self, wd, target_vane_offset=0.):
        """Given the next wind direction step, compute the next step in yaw"""
        # Add target vane offset to perceived wind direction
        wd = wrap_360(wd - target_vane_offset)

        # Update the filtered wind direction using low-pass filtering
        def lpf(x, x_filt):
            return (1 - self.lpf_alpha) * x + self.lpf_alpha * x_filt

        self.wd_cos_filt = lpf(np.cos(np.deg2rad(wd)), self.wd_cos_filt)
        self.wd_sin_filt = lpf(np.sin(np.deg2rad(wd)), self.wd_sin_filt)
        wd_filt = np.rad2deg(np.arctan2(self.wd_sin_filt, self.wd_cos_filt))
        wd_filt = wrap_360(wd_filt)

        # Now get into guts of control
        if self.yaw_state == 1:  # yawing right
            if wrap_180(wd_filt - self.yaw) <= 0:
                self.yaw_state = 0  # Stop yawing
            else:
                self.yaw = wrap_360(self.yaw + self.yaw_rate*self.dt)
                self.yaw_state = 1  # persist
        elif self.yaw_state == -1:  # yawing left
            if wrap_180(wd_filt - self.yaw) >= 0:
                self.yaw_state = 0  # Stop yawing
            else:
                self.yaw = wrap_360(self.yaw - self.yaw_rate*self.dt)
                self.yaw_state = -1  # persist
        else:
            if wrap_180(wd_filt - self.yaw) > self.deadband:
                self.yaw_state = 1  # yaw right
            if wrap_180(wd_filt - self.yaw) < -1 * self.deadband:
                self.yaw_state = -1  # yaw left

        return self.yaw
