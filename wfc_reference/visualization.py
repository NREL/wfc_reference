# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt

import floris.tools as wfct


def visualize_flow_field(fi, yaw_angles, wd, ws, ti=0.06, title=None):
    num_turbs = len(fi.layout_x)
    fi.reinitialize_flow_field(
        wind_direction=wd,
        wind_speed=ws,
        turbulence_intensity=ti
        )
    fi.calculate_wake(yaw_angles=yaw_angles)
    hor_plane = fi.get_hor_plane()
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    wfct.visualization.plot_turbines(
        ax=ax,
        layout_x=fi.layout_x,
        layout_y=fi.layout_y,
        yaw_angles=yaw_angles,
        D=fi.floris.farm.turbines[0].rotor_diameter,
        color='black',
        wind_direction=wd
        )
    ax.set_title(title)
    for ti in range(num_turbs):
        ax.text(fi.layout_x[ti], fi.layout_y[ti], 'WTG_%03d' % ti)

    return fig, ax
