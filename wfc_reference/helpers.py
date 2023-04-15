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

# TODO: This a duplicate of what is already in floris/flasc
def wrap_180(x):
    x = np.where(x <= -180.0, x + 360.0, x)
    x = np.where(x > 180.0, x - 360.0, x)
    return float(x)

class LPF_1():
    '''
    1st order low pass filter object

    inputs:     time_const:  (float) time constant of filter
                length: (int)  number of elements in array to filter
    '''
    def __init__(self,time_const,length=1,sample_period=1,x0=None):
        # Set up low-pass filter for wind direction
        self.Ts = sample_period

        # Set initial condition
        if x0 is None:
            self.x_prev = np.full(length,fill_value=0)
        elif len(x0) != length:
            raise Exception(f'Initial condition (x0 = {x0} is not the same as length ({length}')
        else:
            self.x_prev = x0

        # Set coefficients
        self._set_lpf_1_coeffs(time_const)
        

    def _set_lpf_1_coeffs(self,time_const):
        fc = 1 / (2 * np.pi * time_const)
        self.lpf_alpha = np.exp(-2 * np.pi * self.Ts * fc)

    def step(self,x):
        '''
            Return output of LPF, given the current value (x) and the 
            previous value (x_prev)
        '''
        y = (1 - self.lpf_alpha) * np.array(x) + self.lpf_alpha * np.array(self.x_prev)
        self.x_prev = y
        return y