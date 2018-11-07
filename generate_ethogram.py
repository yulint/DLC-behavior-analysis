import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, rc
import time
from scipy.signal import lfilter

from sklearn.manifold import TSNE

from ethogram_analysis_code import zip_xy, find_distance, sniffing_threshold, theta_btwn_vectors, target_theta, orienting_threshold, combined_nosetail_orienting_threshold, cross_corr_xy, following_threshold, movement

female_nose_interpolated_lfilt_x = np.load('data/female_nose_interpolated_lfilt_x.npy')
female_nose_interpolated_lfilt_y = np.load('data/female_nose_interpolated_lfilt_y.npy')

female_tail_interpolated_lfilt_x = np.load('data/female_tail_interpolated_lfilt_x.npy')
female_tail_interpolated_lfilt_y = np.load('data/female_tail_interpolated_lfilt_y.npy')

female_right_ear_interpolated_lfilt_x = np.load('data/female_right_ear_interpolated_lfilt_x.npy')
female_right_ear_interpolated_lfilt_y = np.load('data/female_right_ear_interpolated_lfilt_y.npy')

female_left_ear_interpolated_lfilt_x = np.load('data/female_left_ear_interpolated_lfilt_x.npy')
female_left_ear_interpolated_lfilt_y = np.load('data/female_left_ear_interpolated_lfilt_y.npy')

female_tail_interpolated_lfilt_x = np.load('data/female_tail_interpolated_lfilt_x.npy')
female_tail_interpolated_lfilt_y = np.load('data/female_tail_interpolated_lfilt_y.npy')

male_nose_interpolated_lfilt_x = np.load('data/male_nose_interpolated_lfilt_x.npy')
male_nose_interpolated_lfilt_y = np.load('data/male_nose_interpolated_lfilt_y.npy')

male_tail_interpolated_lfilt_x = np.load('data/male_tail_interpolated_lfilt_x.npy')
male_tail_interpolated_lfilt_y = np.load('data/male_tail_interpolated_lfilt_y.npy')

male_right_ear_interpolated_lfilt_x = np.load('data/male_right_ear_interpolated_lfilt_x.npy')
male_right_ear_interpolated_lfilt_y= np.load('data/male_right_ear_interpolated_lfilt_y.npy')

male_left_ear_interpolated_lfilt_x = np.load('data/male_left_ear_interpolated_lfilt_x.npy')
male_left_ear_interpolated_lfilt_y = np.load('data/male_left_ear_interpolated_lfilt_y.npy')

male_tail_interpolated_lfilt_x = np.load('data/male_tail_interpolated_lfilt_x.npy')
male_tail_interpolated_lfilt_y = np.load('data/male_tail_interpolated_lfilt_y.npy')


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
female_theta_to_male_nose, female_head_to_male_nose_vector, female_head_dir_vector = target_theta(female_nose_xy, female_left_ear_xy, female_right_ear_xy, male_nose_xy)
female_theta_to_male_tail, female_head_to_male_tail_vector, _ = target_theta(female_nose_xy, female_left_ear_xy, female_right_ear_xy, male_tail_xy)
male_theta_to_female_nose, male_head_to_female_nose_vector, male_head_dir_vector = target_theta(male_nose_xy, male_left_ear_xy, male_right_ear_xy, female_nose_xy)
male_theta_to_female_tail, male_head_to_female_tail_vector, _ = target_theta(male_nose_xy, male_left_ear_xy, male_right_ear_xy, female_tail_xy)

###threshold angles for -> nose and tail, and combine by addition
female_interest,_ ,_ = combined_nosetail_orienting_threshold(female_theta_to_male_nose, female_theta_to_male_tail)
male_interest, _, _ = combined_nosetail_orienting_threshold(male_theta_to_female_nose, male_theta_to_female_tail)

fig, (ax1, ax2) = plt.subplots(nrows =2, figsize=(15,9))
ax1.plot(male_interest)
ax1.set_title("male interest")
ax2.plot(female_interest)
ax2.set_title("female interest")

######################### following: cross correlation between xy coords of male vs female
male_cross_corr = cross_corr_xy(male_nose_xy,female_body_midpt_xy,time_bin = 35)
female_cross_corr = cross_corr_xy(female_nose_xy,male_body_midpt_xy)

male_following = following_threshold(male_cross_corr)
female_following = following_threshold(female_cross_corr)

fig, (ax1, ax2) = plt.subplots(nrows =2, figsize=(15,9))
ax1.set_title("male following")
ax1.plot(male_following)
ax2.set_title("female following")
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
ax1.set_title("sexual_pursuit")

#################################

male_movement = np.array(male_movement)
female_movement = np.array(female_movement)

# only 2D lists, could be recursive for each dimension
def pad_zeros(unpadded, shape):	
	result = np.zeros(shape)
	result[:unpadded.shape[0],:unpadded.shape[1]] = unpadded
	return result
shape = (len(nose_nose_dist),2)
female_movement = pad_zeros(female_movement, shape)
male_movement = pad_zeros(male_movement, shape)

behav_metrics = np.array([
	nose_nose_dist, 
	male_nose_female_tail_dist, 
	male_nose_female_body_dist, 
	female_nose_male_tail_dist, 
	female_nose_male_body_dist,
	female_theta_to_male_nose, 
	female_theta_to_male_tail, 
	female_head_dir_vector[:,0],
	female_head_dir_vector[:,1],
	male_theta_to_female_nose, 
	male_head_to_female_nose_vector[:,0], 
	male_head_dir_vector[:,0],
	male_movement[:,0],
	female_movement[:,0],
	male_head_to_female_nose_vector[:,1], 
	male_head_dir_vector[:,1],
	male_movement[:,1],
	female_movement[:,1]
	])    

# for idx, elem in enumerate(behav_metrics):
# 	if type(elem) is not 'numpy.ndarray':
# 		behav_metrics[idx] = np.array(elem)
# 	print(behav_metrics[idx].shape)

tsne = TSNE(n_components=2).fit_transform(behav_metrics)
plt.figure(figsize=(12,8))
plt.title('t-SNE components')
plt.scatter(tsne[:,0], tsne[:,1])
plt.show()
