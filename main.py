import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, rc
from IPython.display import HTML

from edit_data import  get_x_y_data, get_x_y_data_cleanup, start_value_cleanup, interp_0_coords, apply_filter



#Load data and format
mf_interaction = pd.read_hdf('18_10_29_mf_interaction_leftDeepCut_resnet50_mf_interaction_male218_10_29shuffle1_150000.h5')
mf_interaction = mf_interaction.T

#Copy and paste the name of the scorer from the dataframe above (also find out how to get the infor directly from the dataframe..)
scorer = 'DeepCut_resnet50_mf_interaction_male218_10_29shuffle1_150000'

# the df has a MultiIndex format, this means that you need to use .loc function on the frame with multiple indexes
# you cannot access the data with only the scorer as an index, or some other single index, it will not work
mf_interaction.loc[(scorer, 'male_nose')]

### remove poorly predicted coords
female_nose_raw_x, female_nose_raw_y =  get_x_y_data(mf_interaction, scorer, 'female_nose')

female_nose_0s_x, female_nose_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_nose', 0.98)
female_tail_0s_x, female_tail_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_tail', 0.98)
female_right_ear_0s_x, female_right_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_right_ear', 0.98)
female_left_ear_0s_x, female_left_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_left_ear', 0.98)

male_nose_0s_x, male_nose_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_nose', 0.98)
male_tail_0s_x, male_tail_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_tail', 0.98)
male_right_ear_0s_x, male_right_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_right_ear', 0.98)
male_left_ear_0s_x, male_left_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_left_ear', 0.98)

## interpolate to replace poor predictions
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

#### butter filter on coords 


female_nose_interpolated_lfilt_x = apply_filter( "butter", female_nose_interpolated_x)
female_nose_interpolated_lfilt_y = apply_filter( "butter", female_nose_interpolated_y)

female_tail_interpolated_lfilt_x = apply_filter( "butter", female_tail_interpolated_x)
female_tail_interpolated_lfilt_y = apply_filter( "butter", female_tail_interpolated_y)

female_right_ear_interpolated_lfilt_x = apply_filter( "butter", female_right_ear_interpolated_x)
female_right_ear_interpolated_lfilt_y = apply_filter( "butter", female_right_ear_interpolated_y)

female_left_ear_interpolated_lfilt_x = apply_filter( "butter", female_left_ear_interpolated_x)
female_left_ear_interpolated_lfilt_y = apply_filter( "butter", female_left_ear_interpolated_y)

male_nose_interpolated_lfilt_x = apply_filter( "butter", male_nose_interpolated_x)
male_nose_interpolated_lfilt_y = apply_filter( "butter", male_nose_interpolated_y)

male_tail_interpolated_lfilt_x = apply_filter( "butter", male_tail_interpolated_x)
male_tail_interpolated_lfilt_y = apply_filter( "butter", male_tail_interpolated_y)

male_right_ear_interpolated_lfilt_x = apply_filter( "butter", male_right_ear_interpolated_x)
male_right_ear_interpolated_lfilt_y = apply_filter( "butter", male_right_ear_interpolated_y)

male_left_ear_interpolated_lfilt_x = apply_filter( "butter", male_left_ear_interpolated_x)
male_left_ear_interpolated_lfilt_y = apply_filter( "butter", male_left_ear_interpolated_y)

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


############################



start = 1000
end = 2000
    
fig, axes = plt.subplots(nrows=2,figsize=(15,10))

axes[0].plot(np.arange(end-start), female_nose_raw_x[start:end], color = "black")
axes[1].plot(np.arange(end-start), female_nose_raw_y[start:end], color = "black")

axes[0].plot(np.arange(end-start), female_nose_interpolated_x[start:end], color = "blue")
axes[1].plot(np.arange(end-start), female_nose_interpolated_y[start:end], color = "blue")

axes[0].plot(np.arange(end-start), female_nose_interpolated_lfilt_x[start:end], color = "red")
axes[1].plot(np.arange(end-start), female_nose_interpolated_lfilt_y[start:end], color = "red")

fig, ax1 = plt.subplots(figsize=(10,10))

ax1.plot(female_nose_interpolated_x[start:end], female_nose_interpolated_y[start:end], color = "blue")
ax1.plot(female_nose_interpolated_lfilt_x[start:end], female_nose_interpolated_lfilt_y[start:end], color = "red")


