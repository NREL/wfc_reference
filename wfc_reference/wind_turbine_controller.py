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
import numpy as np
# import pandas as pd

from wfc_reference.yaw_controller import yaw_controller


def wrap_180(x):
    x = np.where(x <= -180.0, x + 360.0, x)
    x = np.where(x > 180.0, x - 360.0, x)
    return float(x)


class wind_turbine_controller():
    def __init__(self,
                 turbine_name,
                 yaw_dict_default,
                 yaw_dict_opt,
                 time_init=None,
                 sampling_time=td(seconds=1)
                 ):
        print(
              'Initializing wind turbine controller for turbine %s.'
              % str(turbine_name)
              )

        # General properties and operating conditions
        self.turbine_name = str(turbine_name)
        self.wind_speed = 8.0  # Placeholder, not used in example
        self.turbulence_intensity = 0.06  # Placeholder, not used in example
        self.status = True
        self.dt = sampling_time
        if time_init is None:
            self.time = td(seconds=0)
        else:
            self.time = time_init

        # Initialize torque controller and variables
        self.generator_torque = 0.0  # Placeholder, not used in example
        self.generator_power = 0.0  # Placeholder, not used in example

        # Initialize pitch controller and variables
        self.pitch_angle = 0.0  # Placeholder, not used in example

        # Ensure yaw_controller sampling time matches wind_turbine_controller
        if yaw_dict_default['dt'] != sampling_time.total_seconds():
            print('yaw_dict_default[\'dt\'] does not match sampling_time! '+\
                'Updating yaw_dict_default.')
            yaw_dict_default['dt'] = sampling_time.total_seconds()

        if yaw_dict_opt['dt'] != sampling_time.total_seconds():
            print('yaw_dict_opt[\'dt\'] does not match sampling_time! '+\
                'Updating yaw_dict_opt.')
            yaw_dict_opt['dt'] = sampling_time.total_seconds()

        # Initialize yaw controller and variables
        self.nacelle_heading = yaw_dict_default['wd_init']
        self.vane_angle = 0.0
        self.yaw_controller = yaw_controller(yaw_dict_default)
        self.yaw_dict_default = yaw_dict_default
        self.yaw_dict_opt = yaw_dict_opt

    def _set_conditions_artificial(
        self,
        vane_angle,
        wind_speed,
        turbulence_intensity,
        status
        ):
        """ Overwrite turbine inflow conditions."""
        self.vane_angle = wrap_180(vane_angle)
        self.wind_speed = wind_speed
        self.turbulence_intensity = turbulence_intensity
        self.status = status

    def _overwrite_vane_measurement(self, new_vane_angle):
        self.vane_angle = wrap_180(new_vane_angle)

    def _overwrite_yaw_params(self, toggle=False):
        if toggle:
            yaw_dict = self.yaw_dict_opt
        else:
            yaw_dict = self.yaw_dict_default

        self.yaw_controller.time_const = yaw_dict['time_const']
        self.yaw_controller.deadband = yaw_dict['deadband']
        self.yaw_controller.yaw_rate = yaw_dict['yaw_rate']

        self.yaw_controller.set_lpf()

    def read_turbine_sensors(self):
        """ Read turbine sensors and return values."""
        measurements = {
            'nacelle_heading': self.nacelle_heading,
            'nacelle_vane_angle': self.vane_angle,
            'wind_speed': self.wind_speed,
            'turbulence_intensity': self.turbulence_intensity,
            'status': self.status,
            'generator_power': self.generator_power,
            'generator_torque': self.generator_torque,
        }
        return measurements

    def step(self, corrected_vane_angle=None,
             target_vane_offset=0., wfc_toggle=False):
        """Take one timestep forward and cycle through controllers."""
        self.time += self.dt

        # Update wind farm controller stuff
        self._overwrite_yaw_params(wfc_toggle)
        if corrected_vane_angle is not None:
            self._overwrite_vane_measurement(corrected_vane_angle)

        # Go through regular turbine controller loop
        self._torqueloop()  # Placeholder, does nothing
        self._pitchloop()  # Placeholder, does nothing
        self._yawloop(target_vane_offset=target_vane_offset)

    def _torqueloop(self):
        """Placeholder for real torque controller"""
        self.generator_torque = 500. * self.wind_speed  # Placeholder, does nothing
        self.generator_power = (    # Placeholder, does nothing
            .5 * 1.225 * self.wind_speed**3. *
            .45 * (.25 * np.pi * 100.**2.)
            ) * np.cos(self.vane_angle * np.pi / 180.) ** 1.88

    def _pitchloop(self):
        """Placeholder for real pitch controller"""
        if self.wind_speed > 10.:
            self.pitch_angle = 4.5  # Placeholder, does nothing
        else:
            self.pitch_angle = 0.0  # Placeholder, does nothing

    def _yawloop(self, target_vane_offset):
        """Basic yaw control logic, placeholder for more detailed code"""
        # Calculate wind direction using yaw and vane signals
        # ... assuming no northing errors in signals
        wind_direction = self.nacelle_heading + self.vane_angle

        # Take one timestep forward and update yaw position
        self.nacelle_heading = self.yaw_controller.compute(
            wind_direction, target_vane_offset
            )

        # Update internal vane angle with updated yaw position,
        # ... assuming no northing errors in signals
        self.vane_angle = wrap_180(wind_direction - self.nacelle_heading)
