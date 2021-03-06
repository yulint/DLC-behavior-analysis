import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, rc
from IPython.display import HTML
import time


### Clean up and interpolate coords

def get_x_y_data(data, scorer, bodypart):
    #get x_y_data
    print('bodypart is: ', bodypart)
    bodypart_data = (data.loc[(scorer, bodypart)])
    
    bodypart_data_x = bodypart_data.loc[('x')]
    bodypart_data_y = bodypart_data.loc[('y')]
    
    return bodypart_data_x, bodypart_data_y
    

def get_x_y_data_cleanup(data, scorer, bodypart, likelihood):
    # sets any value below a particular point to value 0 in x and y, this 0 value can then be used by a later
    #interpolation algorithm
    
    bodypart_data = (data.loc[(scorer, bodypart)])
    
    x_coords = []
    y_coords = []
    
    for index in bodypart_data:
        if bodypart_data.loc['likelihood'][index] > likelihood:
            x_coords.append(bodypart_data.loc['x'][index])
            y_coords.append(bodypart_data.loc['y'][index])
        else:
            x_coords.append(0)
            y_coords.append(0)
            
    return x_coords, y_coords

def start_value_cleanup(coords):
    # This is for when the starting value of the coords == 0; interpolation will not work on these coords until the first 0 
    #is changed. The 0 value is changed to the first non-zero value in the coords lists
    for index, value in enumerate(coords):
        if value > 0:
            start_value = value
            start_index = index
            break

    for x in range(start_index):
        coords[x] = start_value



def interp_0_coords(coords_list):
    #coords_list is one if the outputs of the get_x_y_data = a list of co-ordinate points
    for index, value in enumerate(coords_list):
        if value == 0:
            if coords_list[index-1] > 0:
                value_before = coords_list[index-1]
                interp_start_index = index-1
                #print('interp_start_index: ', interp_start_index)
                #print('interp_start_value: ', value_before)
                #print('')

        if index < len(coords_list)-1:
            if value ==0:
                if coords_list[index+1] > 0:
                    interp_end_index = index+1
                    value_after = coords_list[index+1]
                    #print('interp_end_index: ', interp_end_index)
                    #print('interp_end_value: ', value_after)
                    #print('')

                    #now code to interpolate over the values
                    try:
                        interp_diff_index = interp_end_index - interp_start_index
                    except UnboundLocalError:
                        print('the first value in list is 0, use the function start_value_cleanup to fix')
                        break
                    #print('interp_diff_index is:', interp_diff_index)

                    new_values = np.linspace(value_before, value_after, interp_diff_index)
                    #print(new_values)

                    interp_index = interp_start_index+1
                    for x in range(interp_diff_index):
                        #print('interp_index is:', interp_index)
                        #print('new_value should be:', new_values[x])
                        coords_list[interp_index] = new_values[x]
                        interp_index +=1
        if index == len(coords_list)-1:
            if value ==0:
                for x in range(30):
                    coords_list[index-x] = coords_list[index-30]
                    #print('')
    print('function exiting')
    return(coords_list)


def plot_x_y_coords(x_coords, y_coords, start, end, n_data_plotted, data_index): 
    
    if end == 'end':
        end = len(x_coords)
    
    plt.subplot(3,n_data_plotted,data_index)
    plt.plot(x_coords[start:end])
    plt.subplot(3,n_data_plotted,data_index+n_data_plotted)
    plt.plot(y_coords[start:end])
    plt.subplot(3,n_data_plotted,data_index+ (2*n_data_plotted))
    plt.plot(x_coords[start:end], y_coords[start:end])



##### Look at raw data (see the imperfection due to failures of DLC to consistently predict correctly)

male_left_ear_x_raw, male_left_ear_y_raw = get_x_y_data(mf_interaction, scorer, 'male_left_ear')
male_right_ear_x_raw, male_right_ear_y_raw = get_x_y_data(mf_interaction, scorer, 'male_right_ear')
male_nose_x_raw, male_nose_y_raw = get_x_y_data(mf_interaction, scorer, 'male_nose')
male_tail_x_raw, male_tail_y_raw = get_x_y_data(mf_interaction, scorer, 'male_tail')

female_left_ear_x_raw, female_left_ear_y_raw = get_x_y_data(mf_interaction, scorer, 'female_left_ear')
female_right_ear_x_raw, female_right_ear_y_raw = get_x_y_data(mf_interaction, scorer, 'female_right_ear')
female_nose_x_raw, female_nose_y_raw = get_x_y_data(mf_interaction, scorer, 'female_nose')
female_tail_x_raw, female_tail_y_raw = get_x_y_data(mf_interaction, scorer, 'female_tail')

plt.figure(figsize = (10,10))
#use plotting function to plot the coords and see them combined
plt.title('no interpolaton and no filter', size = 15)
plot_x_y_coords(male_nose_x_raw, male_nose_y_raw, 50, 2000, 1, 1)
plt.tight_layout()

#### Look at data after DLC predicted locations that are < a threshold_confidence are removed and interpolated over, starting by looking at only one bodypart

# this will set all values where DLC gave a predicted location at less than a specified confidence interval to 0
female_nose_0s_x, female_nose_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_nose', 0.98)

female_nose_0s_y

# this will interpolate linearly over all co-ordinates set to 0 in the previous function '0scleanup'
start_value_cleanup(female_nose_0s_x)
start_value_cleanup(female_nose_0s_y)
female_nose_interpolated_x = interp_0_coords(female_nose_0s_x)
female_nose_interpolated_y = interp_0_coords(female_nose_0s_y)

plt.figure(figsize = (15,10))

#plot raw
plt.title('no interpolaton and no filter', size = 15)
plot_x_y_coords(male_nose_x_raw, male_nose_y_raw, 50, 200, 2, 1)
plt.title('0s -> lfilter', size = 15)
plot_x_y_coords(female_nose_interpolated_x, female_nose_interpolated_y, 50, 1000, 2, 2)
plt.title('0s -> interpolation -> lfilter', size = 15)
plt.tight_layout()

#Q: Are there more suitable filters than linear filter?

#### Now can smooth over the data using linear fileter

from scipy.signal import lfilter

n= 20 # the larger n is, the smoother curve will be

nom = [1.0 / n] * n
denom = 1
female_nose_interpolated_lfilt_x = lfilter(nom,denom,female_nose_interpolated_x)
female_nose_interpolated_lfilt_y = lfilter(nom,denom,female_nose_interpolated_y)

#this is data before
female_nose_0s_lfilt_x = lfilter(nom,denom,female_nose_0s_x)
female_nose_0s_lfilt_y = lfilter(nom,denom,female_nose_0s_y)


#Comparing how good the home-made interpolation + lfilter is, compared to non-interpolated + filter

plt.figure(figsize = (13,8))

#plot unfiltered

#plt.title('no interpolaton and no filter', size = 15)
plot_x_y_coords(male_nose_x_raw, male_nose_y_raw, 50, 1000, 3, 1)
#plt.title('0s -> lfilter', size = 15)
plot_x_y_coords(female_nose_0s_lfilt_x, female_nose_0s_lfilt_y, 50, 1000, 3, 2)
#plt.title('0s -> interpolation -> lfilter', size = 15)
plot_x_y_coords(female_nose_interpolated_lfilt_x, female_nose_interpolated_lfilt_y, 50, 1000, 3, 3)



plt.tight_layout()



### Have determined that data looks best after likeihood cleanup, interpolation, and then a linear filter -> do for all the data
if __name__ == "main":    
    female_nose_0s_x, female_nose_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_nose', 0.98)
    female_tail_0s_x, female_tail_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_tail', 0.98)
    female_right_ear_0s_x, female_right_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_right_ear', 0.98)
    female_left_ear_0s_x, female_left_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'female_left_ear', 0.98)
    
    
    male_nose_0s_x, male_nose_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_nose', 0.98)
    male_tail_0s_x, male_tail_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_tail', 0.98)
    male_right_ear_0s_x, male_right_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_right_ear', 0.98)
    male_left_ear_0s_x, male_left_ear_0s_y = get_x_y_data_cleanup(mf_interaction, scorer, 'male_left_ear', 0.98)
    
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
    
    
    
        