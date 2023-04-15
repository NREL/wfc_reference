import os

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp


def round_time(dt=None, date_delta=datetime.timedelta(seconds=1), to='down'):
    """
    Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    from:  http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    epoch = datetime.datetime(1970, 1, 1)
    round_to = date_delta.total_seconds()
    if dt is None:
        dt = datetime.now()
    seconds = (dt - epoch).seconds

    if seconds % round_to == 0 and dt.microsecond == 0:
        rounding = (seconds + round_to / 2) // round_to * round_to
    else:
        if to == 'up':
            # // is a floor division, not a comment on following line (like in javascript):
            rounding = (seconds + dt.microsecond/1000000 + round_to) // round_to * round_to
        elif to == 'down':
            rounding = seconds // round_to * round_to
        else:
            rounding = (seconds + round_to / 2) // round_to * round_to

    # Force precision to microseconds
    dt = dt + datetime.timedelta(0, rounding - seconds, - dt.microsecond)
    dt_rounded = datetime.datetime(
        year=dt.year, month=dt.month, day=dt.day, hour=dt.hour,
        minute=dt.minute, second=dt.second, microsecond=dt.microsecond
    )
    return dt_rounded


def interpolate_timeseries(time_target, time_data, y_data):
    # Convert timestamp arrays to numpy datetime64 format
    time_data = np.array(time_data, dtype=np.datetime64)
    time_target = np.array(time_target, dtype=np.datetime64)

    # Define reference time
    epoch = np.min([time_data[0], time_target[0]])

    # Get integers
    xp = np.array(time_data - epoch, dtype=int)
    x = np.array(time_target - epoch, dtype=int)

    # Interpolate values using nearest-neighbor
    f = interp.interp1d(xp, y_data, kind='nearest', fill_value="extrapolate")
    return f(x)


def interpolate_dataframe(df_in, time_target):
    df_out = pd.DataFrame(
        {'time': time_target}
    )
    cols = [c for c in df_in.columns if 'time' not in c]
    for c in cols:
        df_out[c] = interpolate_timeseries(
            time_target=time_target,
            time_data=df_in['time'],
            y_data=df_in[c]
        )
    return df_out


def plot_dataframe(df_in, ax=None):
    x = df_in['time']
    cols = [c for c in df_in.columns if 'time' not in c]
    if ax is None:
        _, ax = plt.subplots(nrows=len(cols), sharex=True)
    for ii, c in enumerate(cols):
        ax[ii].step(x, df_in[c], label=c)
        ax[ii].grid(True)
        ax[ii].set_ylabel(c)
    ax[ii].set_xlabel('Time')
    return ax


def download_and_read_data(no_files=1):
    from scipy.io import loadmat
    import urllib.request
    import requests
    import re

    # Get all filenames
    url = "https://wind.nrel.gov/MetData/135mData/M5Twr/20Hz/mat/2012/11/07/"
    page = requests.get(url).text
    inds = [
        m.start() for m in re.finditer("[0-9][0-9]_\w+[0-9].mat</a>", page)
    ]
    filenames = [page[i:i+27] for i in inds]
    print("Found %d files on the remote URL page." % len(filenames))

    df_out = []
    root_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_path, 'raw_data_files')
    os.makedirs(out_path, exist_ok=True)
    for ii, fn in enumerate(filenames[0:no_files]):
        fn_path = os.path.join(out_path, fn)
        if os.path.exists(fn_path):
            print("[%02d/%02d] Found local file %s." % (ii, no_files-1, fn))
        else:
            print("[%02d/%02d] Downloading file %s..." % (ii, no_files-1, fn))
            urllib.request.urlretrieve(url + fn, fn_path)

        # Load the .mat file
        try:
            print("...Loading and formatting %s..." % fn)
            data = loadmat(fn_path)
        except:
            # Try redownloading...
            print("...File is corrupt. Redownloading...")
            urllib.request.urlretrieve(url + fn, fn_path)
            data = loadmat(fn_path) 
            print("...File redownloaded and loaded successfully.")

        # Convert the variables to Python
        time = np.array(data["time_UTC"][0][0][0], dtype=float).flatten()
        time = pd.to_datetime(time - 719529, unit="D")
        wind_speed = data["Cup_WS_87m"][0][0][0]
        wind_direction = data["Vane_WD_87m"][0][0][0]

        # Format them to be an array
        wind_speed = np.array(wind_speed, dtype=float).flatten()
        wind_direction = np.array(wind_direction, dtype=float).flatten()

        df_out.append(
            pd.DataFrame(
                {
                    'time': time,
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction
                }
            )
        )

    df_out_concat = pd.concat(df_out).reset_index(drop=True)
    return df_out_concat


if __name__ == "__main__":
    # Load raw data
    df = download_and_read_data(no_files=144)

    # Generate a time array sampled at 1 second
    precision = datetime.timedelta(seconds=1)
    time_array_src = list(df['time'])
    t0 = round_time(dt=time_array_src[0], date_delta=precision, to="down")
    t1 = round_time(dt=time_array_src[-1], date_delta=precision, to="up")
    t = t0
    t_array = []
    while t <= t1:
        t_array.append(t)
        t += precision

    # Interpolate dataframe linearly onto new time array
    df_out = interpolate_dataframe(df, t_array)

    # Plot dataframes
    ax = plot_dataframe(df)
    ax = plot_dataframe(df_out, ax=ax)
    ax[0].legend(['Raw data [20 Hz]', 'Downsampled data [1 Hz]'])
    ax[1].legend(['Raw data [20 Hz]', 'Downsampled data [1 Hz]'])
    plt.show()

    # Save to file
    root_path = os.path.dirname(os.path.abspath(__file__))
    fn_out = os.path.join(root_path, 'example_dataset.ftr')
    df_out.to_feather(fn_out)
    print(df_out)
    print('Saved dataframe as .ftr (feather) file to %s.' % fn_out)
