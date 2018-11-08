import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, rc
from IPython.display import HTML
import time

from edit_data import  get_x_y_data_cleanup, start_value_cleanup, interp_0_coords, apply_filter

def plot_x_y_coords(x_coords, y_coords, start, end, n_data_plotted, data_index): 
    
    if end == 'end':
        end = len(x_coords)
    
    plt.subplot(3,n_data_plotted,data_index)
    plt.plot(x_coords[start:end])
    plt.subplot(3,n_data_plotted,data_index+n_data_plotted)
    plt.plot(y_coords[start:end])
    plt.subplot(3,n_data_plotted,data_index+ (2*n_data_plotted))
    plt.plot(x_coords[start:end], y_coords[start:end])

#Load data and format
mf_interaction = pd.read_hdf('18_10_29_mf_interaction_leftDeepCut_resnet50_mf_interaction_male218_10_29shuffle1_150000.h5')
mf_interaction = mf_interaction.T

#Copy and paste the name of the scorer from the dataframe above (also find out how to get the infor directly from the dataframe..)
scorer = 'DeepCut_resnet50_mf_interaction_male218_10_29shuffle1_150000'

# the df has a MultiIndex format, this means that you need to use .loc function on the frame with multiple indexes
# you cannot access the data with only the scorer as an index, or some other single index, it will not work
mf_interaction.loc[(scorer, 'male_nose')]

### remove poorly predicted coords
female_nose_0s_x, female_nose_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_nose', 0.98)
female_tail_0s_x, female_tail_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_tail', 0.98)
female_right_ear_0s_x, female_right_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_right_ear', 0.98)
female_left_ear_0s_x, female_left_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_left_ear', 0.98)

male_nose_0s_x, male_nose_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_nose', 0.98)
male_tail_0s_x, male_tail_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_tail', 0.98)
male_right_ear_0s_x, male_right_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_right_ear', 0.98)
male_left_ear_0s_x, male_left_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_left_ear', 0.98)

### interpolate to replace poor predictions
start_value_cleanup(male_nose_0s_x)
start_value_cleanup(male_nose_0s_y)
male_nose_interpolated_x = interp_0_coords(male_nose_0s_x)
male_nose_interpolated_y = interp_0_coords(male_nose_0s_y)

start_value_cleanup(male_tail_0s_x)
start_value_cleanup(male_tail_0s_y)
male_tail_interpolated_x = interp_0_coords(male_tail_0s_x)
male_tail_interpolated_y = interp_0_coords(male_tail_0s_y)

start_value_cleanup(male_right_ear_0s_x)
start_value_cleanup(male_right_ear_0s_y)
male_right_ear_interpolated_x = interp_0_coords(male_right_ear_0s_x)
male_right_ear_interpolated_y = interp_0_coords(male_right_ear_0s_y)

start_value_cleanup(male_left_ear_0s_x)
start_value_cleanup(male_left_ear_0s_y)
male_left_ear_interpolated_x = interp_0_coords(male_left_ear_0s_x)
male_left_ear_interpolated_y = interp_0_coords(male_left_ear_0s_y)

start_value_cleanup(female_nose_0s_x)
start_value_cleanup(female_nose_0s_y)
female_nose_interpolated_x = interp_0_coords(female_nose_0s_x)
female_nose_interpolated_y = interp_0_coords(female_nose_0s_y)

start_value_cleanup(female_tail_0s_x)
start_value_cleanup(female_tail_0s_y)
female_tail_interpolated_x = interp_0_coords(female_tail_0s_x)
female_tail_interpolated_y = interp_0_coords(female_tail_0s_y)

start_value_cleanup(female_right_ear_0s_x)
start_value_cleanup(female_right_ear_0s_y)
female_right_ear_interpolated_x = interp_0_coords(female_right_ear_0s_x)
female_right_ear_interpolated_y = interp_0_coords(female_right_ear_0s_y)

start_value_cleanup(female_left_ear_0s_x)
start_value_cleanup(female_left_ear_0s_y)
female_left_ear_interpolated_x = interp_0_coords(female_left_ear_0s_x)
female_left_ear_interpolated_y = interp_0_coords(female_left_ear_0s_y)

#### linear filter on coords 


female_nose_interpolated_lfilt_x = apply_filter( "linear", female_nose_interpolated_x)
female_nose_interpolated_lfilt_y = apply_filter( "linear", female_nose_interpolated_y)

female_tail_interpolated_lfilt_x = apply_filter( "linear", female_tail_interpolated_x)
female_tail_interpolated_lfilt_y = apply_filter( "linear", female_tail_interpolated_y)

female_right_ear_interpolated_lfilt_x = apply_filter( "linear", female_right_ear_interpolated_x)
female_right_ear_interpolated_lfilt_y = apply_filter( "linear", female_right_ear_interpolated_y)

female_left_ear_interpolated_lfilt_x = apply_filter( "linear", female_left_ear_interpolated_x)
female_left_ear_interpolated_lfilt_y = apply_filter( "linear", female_left_ear_interpolated_y)

male_nose_interpolated_lfilt_x = apply_filter( "linear", male_nose_interpolated_x)
male_nose_interpolated_lfilt_y = apply_filter( "linear", male_nose_interpolated_y)

male_tail_interpolated_lfilt_x = apply_filter( "linear", male_tail_interpolated_x)
male_tail_interpolated_lfilt_y = apply_filter( "linear", male_tail_interpolated_y)

male_right_ear_interpolated_lfilt_x = apply_filter( "linear", male_right_ear_interpolated_x)
male_right_ear_interpolated_lfilt_y = apply_filter( "linear", male_right_ear_interpolated_y)

male_left_ear_interpolated_lfilt_x = apply_filter( "linear", male_left_ear_interpolated_x)
male_left_ear_interpolated_lfilt_y = apply_filter( "linear", male_left_ear_interpolated_y)

np.save('data/female_nose_interpolated_lfilt_x', female_nose_interpolated_lfilt_x)
np.save('data/female_nose_interpolated_lfilt_y', female_nose_interpolated_lfilt_y)

np.save('data/female_tail_interpolated_lfilt_x', female_tail_interpolated_lfilt_x)
np.save('data/female_tail_interpolated_lfilt_y', female_tail_interpolated_lfilt_y)

np.save('data/female_right_ear_interpolated_lfilt_x', female_right_ear_interpolated_lfilt_x)
np.save('data/female_right_ear_interpolated_lfilt_y', female_right_ear_interpolated_lfilt_y)

np.save('data/female_left_ear_interpolated_lfilt_x', female_left_ear_interpolated_lfilt_x)
np.save('data/female_left_ear_interpolated_lfilt_y', female_left_ear_interpolated_lfilt_y)

np.save('data/female_tail_interpolated_lfilt_x', female_tail_interpolated_lfilt_x)
np.save('data/female_tail_interpolated_lfilt_y', female_tail_interpolated_lfilt_y)

np.save('data/male_nose_interpolated_lfilt_x', male_nose_interpolated_lfilt_x)
np.save('data/male_nose_interpolated_lfilt_y', male_nose_interpolated_lfilt_y)

np.save('data/male_tail_interpolated_lfilt_x', male_tail_interpolated_lfilt_x)
np.save('data/male_tail_interpolated_lfilt_y', male_tail_interpolated_lfilt_y)

np.save('data/male_right_ear_interpolated_lfilt_x', male_right_ear_interpolated_lfilt_x)
np.save('data/male_right_ear_interpolated_lfilt_y', male_right_ear_interpolated_lfilt_y)

np.save('data/male_left_ear_interpolated_lfilt_x', male_left_ear_interpolated_lfilt_x)
np.save('data/male_left_ear_interpolated_lfilt_y', male_left_ear_interpolated_lfilt_y)

np.save('data/male_tail_interpolated_lfilt_x', male_tail_interpolated_lfilt_x)
np.save('data/male_tail_interpolated_lfilt_y', male_tail_interpolated_lfilt_y)


