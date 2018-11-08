import numpy as np
from scipy.signal import find_peaks 
import matplotlib.pyplot as plt 

def zip_xy(x_pos, y_pos):
    
    xy = np.stack((x_pos, y_pos), axis =-1)
    
    return xy

### for nose-nose/tail/body distances

def find_distance(body_part_1_xy, body_part_2_xy):
    
    vector_btwn_parts = body_part_1_xy - body_part_2_xy
    distanceOverTime = np.linalg.norm(vector_btwn_parts,axis=1)
    #distanceOverTime = np.sqrt(((body_part_1_xy[:,0]-body_part_2_xy[:,0])**2) + ((body_part_1_xy[:,1]-body_part_2_xy[:,1])**2))
    return distanceOverTime

def sniffing_threshold(distanceOverTime, threshold=50):
    sniffing_behaviour = []
    
    for distance in distanceOverTime:
        if distance < threshold:
            sniffing_behaviour.append(1)
        else:
            sniffing_behaviour.append(0)
    
    return sniffing_behaviour

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
        
    return theta_to_target, ear_mid_pt_to_target_vector, head_dir_vector

def orienting_threshold(theta_to_target,threshold = np.pi/12):
    #threshold theta to individual body parts (nose/tail)
    orienting_to_target = []
    
    for theta in theta_to_target:
        if theta < threshold:
            orienting_to_target.append(1)
        else:
            orienting_to_target.append(0)   
            
    return orienting_to_target

def combined_nosetail_orienting_threshold(theta_to_target_nose, theta_to_target_tail, threshold = np.pi/12):
   #combine thresholded theta to nose + tail -> thresholded theta to body
    orienting_to_nose = orienting_threshold(theta_to_target_nose, threshold = threshold)
    orienting_to_tail = orienting_threshold(theta_to_target_tail, threshold = threshold)
    
    interest = np.add(orienting_to_nose, orienting_to_tail)
    
    return interest, orienting_to_nose, orienting_to_tail


### for following

##cross correlation of male vs female xy coordinates, binned by eg seconds
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
        bin_end_2 = (i+2)*time_bin

        cross_corr_x = normalised_cross_corr(body_midpt_xy[bin_start:bin_end_2,0],nose_xy[bin_start:bin_end,0],mode='valid')
        cross_corr_y = normalised_cross_corr(body_midpt_xy[bin_start:bin_end_2,1],nose_xy[bin_start:bin_end,1],mode='valid')
        cross_corr_xy = cross_corr_x + cross_corr_y
        
        peak_idx, _ = find_peaks(cross_corr_xy, prominence = 0.1) 
        
        if len(peak_idx) == 0:
            cross_corr_val = 0
        if len(peak_idx) == 1:
            cross_corr_val = cross_corr_xy[peak_idx]
        if len(peak_idx) >1:
            peaks = []
            for i in peak_idx:
                peaks.append(cross_corr_xy[i])
            cross_corr_val = max(peaks)
        
#        fig, ax1= plt.subplots( figsize=(15,9))
#        ax1.plot(cross_corr_xy)
#        print(peak_idx)        
        cross_corr_over_time = np.append(cross_corr_over_time, np.full(time_bin, cross_corr_val), axis=0)        
        
    return cross_corr_over_time       

def following_threshold(cross_corr_over_time, threshold=0.4):
    following_behaviour = []
    
    for cross_corr in cross_corr_over_time:
        if cross_corr > threshold:
            following_behaviour.append(1)
        else:
            following_behaviour.append(0)
    
    return following_behaviour


### sexual pursuit

#movement vector, where movement is calculated relative to 1 second before
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


####################################
    
def slope(p1, p2):
    return (p2[1] - p1[1]) * 1. / (p2[0] - p1[0])
   
def y_intercept(slope, p1):
    return p1[1] - 1. * slope * p1[0]
   
def intersect(line1, line2):
    min_allowed = 1e-5   # guard against overflow
    big_value = 1e10     # use instead (if overflow would have occurred)
    m1 = slope(line1[0], line1[1])
#     print( 'm1: %d' % m1 )
    b1 = y_intercept(m1, line1[0])
#     print( 'b1: %d' % b1 )
    m2 = slope(line2[0], line2[1])
#     print( 'm2: %d' % m2 )
    b2 = y_intercept(m2, line2[0])
#     print( 'b2: %d' % b2 )
    if abs(m1 - m2) < min_allowed :
        x = big_value
    else :
        x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
#    y2 = m2 * x + b2
#     print( '(x,y,y2) = %d,%d,%d' % (x, y, y2))
    return (int(x),int(y))
 
    
def detect_intersect(line1, line2):
    intersection_pt = intersect(line1, line2)
    
#     print(line1[0][0], line1[1][0], line2[0][0], line2[1][0], intersection_pt[0])
#     print(line1[0][1], line1[1][1], line2[0][1], line2[1][1], intersection_pt[1])
   
    if (line1[0][0] < line1[1][0]) :
        if intersection_pt[0] < line1[0][0] or intersection_pt[0] > line1[1][0]:
#             print( 'exit 1' )
            return 0
    else:
        if intersection_pt[0] > line1[0][0] or intersection_pt[0] < line1[1][0]:
#             print( 'exit 2' )
            return 0
         
    if (line2[0][0] < line2[1][0]) :
        if intersection_pt[0] < line2[0][0] or intersection_pt[0] > line2[1][0]:  
#             print( 'exit 3' )
            return 0
    else:
        if intersection_pt[0] > line2[0][0] or intersection_pt[0] < line2[1][0]:
#             print( 'exit 4' )
            return 0
    
    return 1 



