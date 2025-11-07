import os
import numpy as np

from py_replay_bg.sensors.Vettoretti19CGM import Vettoretti19CGM
from utils import load_test_data, load_patient_info

from py_replay_bg.py_replay_bg import ReplayBG
from py_replay_bg.visualizer import Visualizer
from py_replay_bg.analyzer import Analyzer
import pandas as pd

# Set verbosity
verbose = True
plot_mode = False

# Set other parameters for twinning
blueprint = 'multi-meal'
save_folder = os.path.join(os.path.abspath(''), '..', '..', '..')

# load patient_info
patient_info = load_patient_info()
p = np.where(patient_info['patient'] == 1)[0][0]
# Set bw and u2ss
bw = float(patient_info.bw.values[p])

# Instantiate ReplayBG
rbg = ReplayBG(blueprint=blueprint, save_folder=save_folder,
               yts=5, exercise=True,
               seed=1,
               verbose=verbose, plot_mode=plot_mode)

# Load data and set save_name
data = load_test_data(day=1)
save_name = 'data_day_' + str(1)

print("Replaying " + save_name)

# create hr_df with same timestamps and same length as data.t
start = pd.to_datetime(data.t.iloc[0])
end = pd.to_datetime(data.t.iloc[-1])

# minutes between start and end
delta_minutes = int((end - start) / np.timedelta64(1, 'm'))

# target length following the same formula used for self.t
target_len = delta_minutes + 5

# create 1-minute timestamps starting at start with target_len rows
idx = pd.date_range(start=start, periods=target_len, freq='1min')

hr_df = pd.DataFrame({'t': idx, 'hr': 60})
hr_df['hr'] = 60

hr_df.loc[400:440, 'hr'] = 120
hr_df.loc[800:860, 'hr'] = 100
hr_df.loc[900:960, 'hr'] = 100
hr_df.loc[1000:1060, 'hr'] = 50

# Replay the twin with the same input data used for twinning
replay_results = rbg.replay(data=data, bw=bw, save_name=save_name, hr = hr_df,
                            twinning_method='map',
                            save_workspace=False,
                            save_suffix='_replay_map')

# Visualize and analyze results
Visualizer.plot_replay_results(replay_results, data=data)
