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

from floris.tools import floris_interface as wfct

from flasc import floris_tools as ftools


def load_floris():
    root_path = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(root_path, "example_floris_input.yaml")
    fi = wfct.FlorisInterface(fn)  # Load FLORIS object
    return fi


if __name__ == "__main__":
    # Set up FLORIS model
    fi = load_floris()
    num_turbs = len(fi.layout_x)

    # Get upstream turbines as a function of wind direction
    df_upstream = ftools.get_upstream_turbs_floris(fi)
    print(df_upstream)

    # Save to a .csv file
    root_path = os.path.dirname(os.path.abspath(__file__))
    df_upstream.to_csv(
        os.path.join(root_path, 'df_upstream.csv'), index=False
    )
