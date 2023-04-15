# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from itertools import product 

from matplotlib import pyplot as plt
import numpy as np


class spline_approximator():
    # Initialize class
    def __init__(self, wd_array, yaw_angles, verbose=True, debug=False):
        # Save information about yaw curve to self
        self.wd_array = wd_array
        self.yaw_angles = yaw_angles
        self.yaw_lb = np.min(yaw_angles)
        self.yaw_ub = np.max(yaw_angles)

        # Other toolbox settings
        self.debug = debug
        self.verbose = verbose

        # Initialize approximant as true solution
        self._set_approximant(wd_array, yaw_angles)

    def _set_approximant(self, x, y):
        self.spline_x = np.array(x, dtype=float)
        self.spline_y = np.array(y, dtype=float)
        self.spline_N = len(x)

    def _get_approximant(self):
        return np.array(self.spline_x, copy=True), np.array(self.spline_y, copy=True)

    def apply_simple_spline_approximation(self, max_error=0.10, max_rmse=0.10):
        # Import options
        verbose = self.verbose
        debug = self.debug

        # Start with current approximation
        wd_array, yaw_angles = self._get_approximant()

        # Initialize array to save x-coordinates of the spline
        segment_ids = [0]  # Initial condition: always include left-most point
        ii_rb_opt = 0  # Initial condition
        while ii_rb_opt < len(wd_array) - 1:
            ii_lb = ii_rb_opt
            wd_lb = wd_array[ii_lb]
            yaw_lb = yaw_angles[ii_lb]
            if debug:
                print('Left bound, wd_lb={}, yaw_lb={}'.format(wd_lb, yaw_lb))
            bool_array = []
            ii_rb_eval_array = np.array(range(ii_lb + 1, len(wd_array)), dtype=int)
            for ii_rb in ii_rb_eval_array:
                wd_rb = wd_array[ii_rb]
                yaw_rb = yaw_angles[ii_rb]
                wd_subset = wd_array[ii_lb:ii_rb]
                yaw_subset = yaw_angles[ii_lb:ii_rb]
                yaw_subset_approx = np.interp(wd_subset, [wd_lb, wd_rb], [yaw_lb, yaw_rb])
                error_curve = yaw_subset - yaw_subset_approx
                rmse = np.sqrt(np.mean(error_curve**2.0))
                largest_error = np.max(np.abs(error_curve))
                acceptable = (rmse < max_rmse) & (largest_error < max_error)
                bool_array.append(acceptable)
                if debug:
                    print("ii_rb={}, wd_rb={}, yaw_rb={}, rmse={}, max_error={}. Acceptability={}".format(ii_rb, wd_rb, yaw_rb, rmse, max_error, acceptable))

            opt_id = np.where(bool_array)[0][-1]
            ii_rb_opt = ii_rb_eval_array[opt_id]
            segment_ids.append(ii_rb_opt)
            if debug:
                print("Optimal ii_rb to add as a marker: {}".format(ii_rb_opt))

        if debug:
            print("Segment IDs: {}".format(segment_ids))
        
        # Set approximant
        x_spline = np.array(wd_array)[segment_ids]
        y_spline = np.array(yaw_angles)[segment_ids]
        self._set_approximant(x_spline, y_spline)
        if verbose:
            print("Simple spline approximant (N={:d}). Please reduce order accordingly using 'apply_spline_order_reduction()'.".format(self.spline_N))
        return self._get_approximant()

    def _reduce_spline_by_one_order(
        self,
        explore_dx=np.arange(-5.0, 5.001, 0.5),  # Degrees of shift in spline point to explore
        explore_n=1,  # Number of spline points to explore the cost function with to left and right of removed spline point
    ):
        # Initialize the true values as the non-approximated ones
        x_full = np.array(self.wd_array, copy=True)
        y_full = np.array(self.yaw_angles, copy=True)

        # Initialize initial solution as the current approximation
        x_spline_init, y_spline_init = self._get_approximant()

        # Define a cost function
        def get_cost(xs, ys):
            return np.sqrt(np.mean((np.interp(x_full, xs, ys) - y_full)**2.0))

        # This function finds the coordinate to remove, and the other coordinates
        # to displace, for a minimal-error reduction in spline dimensionality
        J_init = get_cost(x_spline_init, y_spline_init)  # Initial cost

        # Find all permutations to explore
        ids_to_explore = np.arange(-explore_n, explore_n, 1, dtype=int)
        X = list(product(*[list(explore_dx)] * (2 * explore_n)))
        X = np.array(X, dtype=float)
        Nx = len(X)
        Ns = len(x_spline_init)
        if self.debug:
            print("Evaluating the cost function {:d} times ({:d} points to consider for removal x {:d} permutations for displacement of neighboring points)".format(Nx * (Ns-2), Ns-2, Nx))

        J_opt = 1e9  # Initial cost
        for ii in range(1, Ns - 1):
            if self.debug:
                print("Removing point id {:d} from spline and refitting curve.".format(ii))
            xs_nom = np.hstack([x_spline_init[0:ii], x_spline_init[ii+1::]])  # nominal
            ys_nom = np.hstack([y_spline_init[0:ii], y_spline_init[ii+1::]])  # nominal

            # Remove left-most and right-most bounds from evaluation
            ids = ids_to_explore + ii
            remove_ids = np.array((ids <= 0) | (ids >= len(xs_nom) - 1), dtype=bool)
            ids_to_eval = ids[~remove_ids]
            Xe = [x[~remove_ids] for x in X if np.all(x[remove_ids] == 0)]

            if self.debug:
                print("Exploring {:d} permutations for removing this ID.".format(len(Xe)))
            for xr in Xe:
                if self.debug:
                    print("evaluating xr: {}".format(xr))
                xs = np.array(xs_nom, dtype=float)  # copy
                xs[ids] += xr  # shift x position around
                if not np.all(np.diff(xs) > 0):
                    if self.debug:
                        print("xs is not sorted. Continuing")
                    continue  # If no longer sorted (point shifted too far to the left/right that it crossed another point), skip evaluation

                ys = np.interp(xs, xs_nom, ys_nom)
                J = get_cost(xs, ys)
                if self.debug:
                    print(" Cost for xr={} is J={}".format(xr, J))
                if J < J_opt:
                    if self.debug:
                        print("  Found new optimal solution with xr={}, Jopt={}".format(xr, J_opt))
                    J_opt = J
                    xs_opt = xs
                    ys_opt = ys

        # Finally save the optimal solution to self
        self._set_approximant(xs_opt, ys_opt)
        if self.verbose:
            print("Reduced dimension of spline from {:d} to {:d}. Change in RMSE: {:.2f} %.".format(len(xs_opt) + 1, len(xs_opt), 100 * (J_opt - J_init) / J_init))
        
        return self._get_approximant()

    def _polish_spline(
        self,
        explore_dx=np.arange(-5.0, 5.001, 0.1),
        explore_dy=np.arange(-1.0, 1.001, 0.1),
    ):
        # Initialize the true values as the non-approximated ones
        x_full = np.array(self.wd_array, copy=True)
        y_full = np.array(self.yaw_angles, copy=True)

        # Initialize initial solution as the current approximation
        xs_opt, ys_opt = self._get_approximant()

        # Define cost function
        def get_cost(xs, ys):
            return np.sqrt(np.mean((np.interp(x_full, xs, ys) - y_full)**2.0))
        
        Ns = len(xs_opt)
        J_init = get_cost(xs_opt, ys_opt)
        J_opt = J_init
        for id in range(1, Ns - 1):
            for dx in explore_dx:
                xs_eval = np.array(xs_opt, copy=True)
                xs_eval[id] += dx
                if not np.all(np.diff(xs_eval) > 0):
                    continue  # Jumped over another point, skip
                for dy in explore_dy:
                    ys_eval = np.array(ys_opt, copy=True)
                    ys_eval[id] += dy
                    if ((ys_eval[id] < self.yaw_lb) or (ys_eval[id] > self.yaw_ub)):
                        continue  # Only evaluate points within bounds
                    J = get_cost(xs_eval, ys_eval)
                    
                    if J < J_opt:
                        if self.debug:
                            print("Found new optimal solution with dx={}, dy={}.".format(dx, dy))
                        J_opt = J
                        xs_opt = np.array(xs_eval, dtype=float, copy=True)
                        ys_opt = np.array(ys_eval, dtype=float, copy=True)
        
        self._set_approximant(xs_opt, ys_opt)
        
        if self.verbose:
            print("Polished spline with N={:d}. Change in RMSE: {:.2f} %.".format(self.spline_N, 100 * (J_opt - J_init) / J_init))

        return self._get_approximant()


    def apply_spline_order_reduction(
        self,
        desired_spline_order=20,
        explore_dx=np.arange(-5.0, 5.001, 0.5),
        explore_n=1,
        polish_final_solution=True,
        polish_explore_dx=np.arange(-5.0, 5.001, 0.1), 
        polish_explore_dy=np.arange(-1.0, 1.001, 0.1),
    ):
        # Keep reducing spline by one order until desired order is achieved
        while self.spline_N > desired_spline_order:
            self._reduce_spline_by_one_order(explore_dx, explore_n)

        # Polish final solution, if necessary
        if polish_final_solution:
            self._polish_spline(polish_explore_dx, polish_explore_dy)

        # Return optimal spline solution to user
        return self._get_approximant()

    def plot(self):
        # Extract variables from self
        wd_array = self.wd_array
        yaw_angles = self.yaw_angles
        spline_x = self.spline_x
        spline_y = self.spline_y
        spline_N = self.spline_N
        N = len(yaw_angles)

        # Plot this turbine's yaw curves
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wd_array, yaw_angles * 0.0, '--', color="gray")
        ax.plot(wd_array, yaw_angles, '-o', color='black', markersize=3, label="Original curve (N={:d})".format(N))
        ax.plot(spline_x, spline_y, '-o', color='tab:red', markersize=4, label="Spline approximant (N={:d})".format(spline_N))
        ax.grid(True)
        ax.set_xlabel("Wind direction (deg)")
        ax.set_ylabel("Yaw offset angle (deg)")
        b = (0.5, 1.10)
        ax.legend(loc='upper center', bbox_to_anchor=b, ncol=3, fancybox=False, framealpha=0)
        ax.set_xlim([wd_array[0], wd_array[-1]])
        ax.set_ylim([np.min(yaw_angles) - 5.0, np.max(yaw_angles) + 5.0])
        plt.tight_layout()
        return fig, ax
