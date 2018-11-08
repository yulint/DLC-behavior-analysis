import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation, rc
from IPython.display import HTML
import time
from scipy.signal import butter, filtfilt, lfilter


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

def apply_filter(filter_type, vec):

    if filter_type is 'linear':
        filt_size = 5
        nom = [1.0 / filt_size] * filt_size
        denom = 1
        output = filtfilt(nom,denom,vec)

    if filter_type is 'butter':
        b, a = butter(6, 0.5)
        output = filtfilt(b, a, vec)

    return output
