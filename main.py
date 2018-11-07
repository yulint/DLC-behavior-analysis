import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, rc
from IPython.display import HTML
import time
from scipy.signal import filtfilt, butter, lfilter

from edit_data import  get_x_y_data_cleanup, start_value_cleanup, interp_0_coords
from ethogram_analysis_code import zip_xy, find_distance, sniffing_threshold, theta_btwn_vectors, target_theta, orienting_threshold, combined_nosetail_orienting_threshold, cross_corr_xy, following_threshold, movement


#Load data and format
mf_interaction = pd.read_hdf('18_10_29_mf_interaction_leftDeepCut_resnet50_mf_interaction_male218_10_29shuffle1_150000.h5')
#mf_interaction_female = pd.read_hdf('23_10_18_mf2_interaction1DeepCut_resnet50_mf_interaction_female18_10_23shuffle1_150000.h5')
mf_interaction = mf_interaction.T
#mf_interaction_female = mf_interaction_female.T

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
`

female_nose_interpolated_lfilt_x = lfilter(nom,denom,female_nose_interpolated_x)
female_nose_interpolated_lfilt_y = lfilter(nom,denom,female_nose_interpolated_y)

female_tail_interpolated_lfilt_x = lfilter(nom,denom,female_tail_interpolated_x)
female_tail_interpolated_lfilt_y = lfilter(nom,denom,female_tail_interpolated_y)

female_right_ear_interpolated_lfilt_x = lfilter(nom,denom,female_right_ear_interpolated_x)
female_right_ear_interpolated_lfilt_y = lfilter(nom,denom,female_right_ear_interpolated_y)

female_left_ear_interpolated_lfilt_x = lfilter(nom,denom,female_left_ear_interpolated_x)
female_left_ear_interpolated_lfilt_y = lfilter(nom,denom,female_left_ear_interpolated_y)

male_nose_interpolated_lfilt_x = lfilter(nom,denom,male_nose_interpolated_x)
male_nose_interpolated_lfilt_y = lfilter(nom,denom,male_nose_interpolated_y)

male_tail_interpolated_lfilt_x = lfilter(nom,denom,male_tail_interpolated_x)
male_tail_interpolated_lfilt_y = lfilter(nom,denom,male_tail_interpolated_y)

male_right_ear_interpolated_lfilt_x = lfilter(nom,denom,male_right_ear_interpolated_x)
male_right_ear_interpolated_lfilt_y = lfilter(nom,denom,male_right_ear_interpolated_y)

male_left_ear_interpolated_lfilt_x = lfilter(nom,denom,male_left_ear_interpolated_x)
male_left_ear_interpolated_lfilt_y = lfilter(nom,denom,male_left_ear_interpolated_y)


##################################################################################################################
###ethogram analysis###

####combine into array of xy vectors over time ####

female_nose_xy = zip_xy(female_nose_interpolated_lfilt_x,female_nose_interpolated_lfilt_y)
female_tail_xy = zip_xy(female_tail_interpolated_lfilt_x,female_tail_interpolated_lfilt_y)
female_right_ear_xy = zip_xy(female_right_ear_interpolated_lfilt_x,female_right_ear_interpolated_lfilt_y)
female_left_ear_xy = zip_xy(female_left_ear_interpolated_lfilt_x,female_left_ear_interpolated_lfilt_y)

female_body_midpt_xy = (female_nose_xy + female_tail_xy)/2

male_nose_xy = zip_xy(male_nose_interpolated_lfilt_x,male_nose_interpolated_lfilt_y)
male_tail_xy = zip_xy(male_tail_interpolated_lfilt_x,male_tail_interpolated_lfilt_y)
male_right_ear_xy = zip_xy(male_right_ear_interpolated_lfilt_x,male_right_ear_interpolated_lfilt_y)
male_left_ear_xy = zip_xy(male_left_ear_interpolated_lfilt_x,male_left_ear_interpolated_lfilt_y)

male_body_midpt_xy = (male_nose_xy + male_tail_xy)/2

################### calculate distances between points ####
nose_nose_dist = find_distance(male_nose_xy, female_nose_xy)
male_nose_female_tail_dist  = find_distance(male_nose_xy, female_tail_xy)
male_nose_female_body_dist  = find_distance(male_nose_xy, female_body_midpt_xy)

female_nose_male_tail_dist  = find_distance(female_nose_xy, male_tail_xy)
female_nose_male_body_dist  = find_distance(female_nose_xy, male_body_midpt_xy)

################### threshold distances for sniffing behaviour
mutual_sniffing = sniffing_threshold(nose_nose_dist)

male_anogenital_sniffing = sniffing_threshold(male_nose_female_tail_dist)
male_body_sniffing = sniffing_threshold(male_nose_female_body_dist)

female_anogenital_sniffing = sniffing_threshold(female_nose_male_tail_dist)
female_body_sniffing = sniffing_threshold(female_nose_male_body_dist)

fig, axes = plt.subplots(nrows=3, sharex = True, figsize=(15,9))

axes[0].plot(mutual_sniffing)
axes[0].set_title("mutual sniffing (nose-nose)")

axes[1].plot(male_anogenital_sniffing, color = 'red')
axes[1].plot(male_body_sniffing, color = 'blue')
axes[1].set_title("male anogenital(red)/ body(blue) sniffing")

axes[2].plot(female_anogenital_sniffing, color = 'red')
axes[2].plot(female_body_sniffing, color = 'blue')
axes[2].set_title("female anogenital(red)/ body(blue) sniffing")

######################### orienting

###### angle btwn mouse 1 head dir and mouse 2, and vector of midpt of ears of mouse 1 -> nose/ tail of mouse 2
female_theta_to_male_nose, _ = target_theta(female_nose_xy, female_left_ear_xy, female_right_ear_xy, male_nose_xy)
female_theta_to_male_tail, _ = target_theta(female_nose_xy, female_left_ear_xy, female_right_ear_xy, male_tail_xy)
male_theta_to_female_nose, male_head_to_female_nose_vector = target_theta(male_nose_xy, male_left_ear_xy, male_right_ear_xy, female_nose_xy)
male_theta_to_female_tail, male_head_to_female_tail_vector = target_theta(male_nose_xy, male_left_ear_xy, male_right_ear_xy, female_tail_xy)

###threshold angles for -> nose and tail, and combine by addition
female_interest,_ ,_ = combined_nosetail_orienting_threshold(female_theta_to_male_nose, female_theta_to_male_tail)
male_interest, _, _ = combined_nosetail_orienting_threshold(male_theta_to_female_nose, male_theta_to_female_tail)

fig, (ax1, ax2) = plt.subplots(nrows =2, figsize=(15,9))
ax1.plot(male_interest)
ax2.plot(female_interest)

######################### following: cross correlation between xy coords of male vs female
male_cross_corr = cross_corr_xy(male_nose_xy,female_body_midpt_xy,time_bin = 35)
female_cross_corr = cross_corr_xy(female_nose_xy,male_body_midpt_xy)

male_following = following_threshold(male_cross_corr)
female_following = following_threshold(female_cross_corr)

fig, (ax1, ax2) = plt.subplots(nrows =2, figsize=(15,9))
ax1.plot(male_following)
ax2.plot(female_following)

########################## sexual pursuit

male_movement, _, _ = movement(male_nose_xy)
female_movement, _, _ = movement(female_nose_xy)

theta_btwn_movement = theta_btwn_vectors(male_movement,female_movement)
theta_movement_to_female_nose = theta_btwn_vectors(male_movement, male_head_to_female_nose_vector[:-1,])
theta_movement_to_female_tail = theta_btwn_vectors(male_movement, male_head_to_female_tail_vector[:-1,])

#male and female have to move in same direction
coherent_motion = orienting_threshold(theta_btwn_movement, threshold = np.pi/12)

#and male has to move towards female
movement_towards_female, _,_ = combined_nosetail_orienting_threshold(theta_movement_to_female_nose, theta_movement_to_female_nose, threshold = np.pi/12)

sexual_pursuit = np.multiply(coherent_motion, movement_towards_female)

fig, ax1 = plt.subplots(figsize=(15,4))
ax1.plot(sexual_pursuit)