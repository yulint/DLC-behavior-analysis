import scipy
import numpy as np
import matplotlib.pyplot as plt

def zip_xy(x_pos, y_pos):
    
    xy = np.stack((x_pos, y_pos), axis =-1)
    
    return xy

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

### for nose-nose/tail/body distances

def find_distance(body_part_1_xy, body_part_2_xy):
    
    vector_btwn_parts = body_part_1_xy - body_part_2_xy
    distanceOverTime = np.linalg.norm(vector_btwn_parts,axis=1)
    #distanceOverTime = np.sqrt(((body_part_1_xy[:,0]-body_part_2_xy[:,0])**2) + ((body_part_1_xy[:,1]-body_part_2_xy[:,1])**2))
    return distanceOverTime

nose_nose_dist = find_distance(male_nose_xy, female_nose_xy)
male_nose_female_tail_dist  = find_distance(male_nose_xy, female_tail_xy)
male_nose_female_body_dist  = find_distance(male_nose_xy, female_body_midpt_xy)

female_nose_male_tail_dist  = find_distance(female_nose_xy, male_tail_xy)
female_nose_male_body_dist  = find_distance(female_nose_xy, male_body_midpt_xy)

def sniffing_threshold(distanceOverTime, threshold=50):
    sniffing_behaviour = []
    
    for distance in distanceOverTime:
        if distance < threshold:
            sniffing_behaviour.append(1)
        else:
            sniffing_behaviour.append(0)
    
    return sniffing_behaviour

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

### orienting, male/female "interest"

def theta_btwn_vectors(vector1, vector2):
    #note that this code confuses thetas < pi/2 with thetas > 3pi/2, and similarly for lower 2 quadrants    
    theta_btwn_vectors = []
    
    for t in range(len(vector1)):
        dot_pdt = np.dot(vector1[t],vector2[t])
        
        theta = np.arccos(dot_pdt/np.linalg.norm(vector1[t])/np.linalg.norm(vector2[t]))
        theta_btwn_vectors.append(theta)
        
    return theta_btwn_vectors

def target_theta(nose_xy, left_ear_xy, right_ear_xy, target_xy):
#theta between middle of head of mouse 1 to target body part of mouse 2    
    mid_pt_btwn_ears_xy = (left_ear_xy+right_ear_xy)/2
    
    head_dir_vector = nose_xy - mid_pt_btwn_ears_xy
      
    ear_mid_pt_to_target_vector =  target_xy - mid_pt_btwn_ears_xy
    
    theta_to_target = theta_btwn_vectors(head_dir_vector,ear_mid_pt_to_target_vector)
        
    return theta_to_target, ear_mid_pt_to_target_vector

def orienting_threshold(theta_to_target,threshold = np.pi/12):
    #threshold theta to individual body parts (nose/tail)
    orienting_to_target = []
    
    for theta in theta_to_target:
        if theta < threshold:
            orienting_to_target.append(1)
        else:
            orienting_to_target.append(0)   
            
    return orienting_to_target

def combined_nosetail_orienting(theta_to_target_nose, theta_to_target_tail, threshold = np.pi/12):
   #combine thresholded theta to nose + tail -> thresholded theta to body
    orienting_to_nose = orienting_threshold(theta_to_target_nose, threshold = threshold)
    orienting_to_tail = orienting_threshold(theta_to_target_tail, threshold = threshold)
    
    interest = np.add(orienting_to_nose, orienting_to_tail)
    
    return interest, orienting_to_nose, orienting_to_tail

female_theta_to_male_nose, _ = target_theta(female_nose_xy, female_left_ear_xy, female_right_ear_xy, male_nose_xy)
female_theta_to_male_tail, _ = target_theta(female_nose_xy, female_left_ear_xy, female_right_ear_xy, male_tail_xy)
male_theta_to_female_nose, male_head_to_female_nose_vector = target_theta(male_nose_xy, male_left_ear_xy, male_right_ear_xy, female_nose_xy)
male_theta_to_female_tail, male_head_to_female_tail_vector = target_theta(male_nose_xy, male_left_ear_xy, male_right_ear_xy, female_tail_xy)

female_interest,_ ,_ = combined_nosetail_orienting(female_theta_to_male_nose, female_theta_to_male_tail)
male_interest, _, _ = combined_nosetail_orienting(male_theta_to_female_nose, male_theta_to_female_tail)

fig, (ax1, ax2) = plt.subplots(nrows =2, figsize=(15,9))
ax1.plot(male_interest)
ax2.plot(female_interest)

### following

def normalised_cross_corr(a,v, mode='valid'):
    a_norm = (a - np.mean(a)) / (np.std(a) * len(a))
    v_norm = (v - np.mean(v)) /  np.std(v)
    cross_corr = np.correlate(a_norm,v_norm, mode=mode)
    return cross_corr

def cross_corr_xy(nose_xy, body_midpt_xy,time_bin = 35):
    num_bins = int(len(nose_xy)/time_bin)
    cross_corr_over_time = []

    for i in range(num_bins):
        bin_start = i*time_bin
        bin_end = (i+1)*time_bin
        bin_end_2 = (i+3)*time_bin

        cross_corr_x = normalised_cross_corr(body_midpt_xy[bin_start:bin_end_2,0],nose_xy[bin_start:bin_end,0],mode='valid')
        cross_corr_y = normalised_cross_corr(body_midpt_xy[bin_start:bin_end_2,1],nose_xy[bin_start:bin_end,1],mode='valid')
        cross_corr_xy = cross_corr_x * cross_corr_y
        
        cross_corr_over_time = np.append(cross_corr_over_time, np.full(time_bin,np.amax(cross_corr_xy)), axis=0)        
        
    return cross_corr_over_time       

def following_threshold(cross_corr_over_time, threshold=0.07):
    following_behaviour = []
    
    for cross_corr in cross_corr_over_time:
        if cross_corr > threshold:
            following_behaviour.append(1)
        else:
            following_behaviour.append(0)
    
    return following_behaviour

male_cross_corr = cross_corr_xy(male_nose_xy,female_body_midpt_xy,time_bin = 35)
female_cross_corr = cross_corr_xy(female_nose_xy,male_body_midpt_xy)

male_following = following_threshold(male_cross_corr)
female_following = following_threshold(female_cross_corr)

fig, (ax1, ax2) = plt.subplots(nrows =2, figsize=(15,9))
ax1.plot(male_following)
ax2.plot(female_following)

### sexual pursuit

def movement(xy, time_scale=35):
    
    movement_vector = []
    movement_velocity = [] #i.e. velocity
    movement_direction= [] #i.e. direction
    
    for t in np.arange(35,len(xy)-1):
        movement = xy[t] - xy[t-time_scale]
        magnitude = np.linalg.norm(movement)
        unit_vector = movement/ np.linalg.norm(movement)
        
        movement_vector.append(movement)
        movement_velocity.append(magnitude)
        movement_direction.append(unit_vector)
    
    return movement_vector, movement_velocity, movement_direction

male_movement, _, _ = movement(male_nose_xy)
female_movement, _, _ = movement(female_nose_xy)

theta_btwn_movement = theta_btwn_vectors(male_movement,female_movement)
theta_movement_to_female_nose = theta_btwn_vectors(male_movement, male_head_to_female_nose_vector[:-1,])
theta_movement_to_female_tail = theta_btwn_vectors(male_movement, male_head_to_female_tail_vector[:-1,])

#movement of male and female in same direction
coherent_motion = orienting_threshold(theta_btwn_movement, threshold = np.pi/12)

#and male moving towards female
movement_towards_female, _,_ = combined_nosetail_orienting(theta_movement_to_female_nose, theta_movement_to_female_nose, threshold = np.pi/12)

sexual_pursuit = np.multiply(coherent_motion, movement_towards_female)

fig, ax1 = plt.subplots(figsize=(15,4))
ax1.plot(sexual_pursuit)