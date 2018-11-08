# annotate video 

import numpy as np 
import cv2
import os 
import pickle
import matplotlib.cm as cm

# only keep long sequences of behaviours 
# window size is minimum length of behaviour
# behaviours lasting less will be filtered

#START_POINT = 9362
START_POINT = 9362
END_POINT =  10000 # 16718

def load_data_dict():
    
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

    data = [female_nose_interpolated_lfilt_x,
    female_nose_interpolated_lfilt_y,
    female_tail_interpolated_lfilt_x,
    female_tail_interpolated_lfilt_y,
    female_right_ear_interpolated_lfilt_x,
    female_right_ear_interpolated_lfilt_y,
    female_left_ear_interpolated_lfilt_x,
    female_left_ear_interpolated_lfilt_y,
    female_tail_interpolated_lfilt_x,
    female_tail_interpolated_lfilt_y,
    male_nose_interpolated_lfilt_x,
    male_nose_interpolated_lfilt_y,
    male_tail_interpolated_lfilt_x,
    male_tail_interpolated_lfilt_y,
    male_right_ear_interpolated_lfilt_x,
    male_right_ear_interpolated_lfilt_y,
    male_left_ear_interpolated_lfilt_x,
    male_left_ear_interpolated_lfilt_y,
    male_tail_interpolated_lfilt_x,
    male_tail_interpolated_lfilt_y]

    data_names = ['female_nose_interpolated_lfilt_x',
    'female_nose_interpolated_lfilt_y',
    'female_tail_interpolated_lfilt_x',
    'female_tail_interpolated_lfilt_y',
    'female_right_ear_interpolated_lfilt_x',
    'female_right_ear_interpolated_lfilt_y',
    'female_left_ear_interpolated_lfilt_x',
    'female_left_ear_interpolated_lfilt_y',
    'female_tail_interpolated_lfilt_x',
    'female_tail_interpolated_lfilt_y',
    'male_nose_interpolated_lfilt_x',
    'male_nose_interpolated_lfilt_y',
    'male_tail_interpolated_lfilt_x',
    'male_tail_interpolated_lfilt_y',
    'male_right_ear_interpolated_lfilt_x',
    'male_right_ear_interpolated_lfilt_y',
    'male_left_ear_interpolated_lfilt_x',
    'male_left_ear_interpolated_lfilt_y',
    'male_tail_interpolated_lfilt_x',
    'male_tail_interpolated_lfilt_y'] 

    return data, data_names

def dict_list_to_numpy(d):
    for key in d:
        d[key] = np.array(d[key])

def filter_binary(binary_vector, window_size):
    assert window_size > 0, "Window size must be greater than 0"

    num_behaviours = 0 
    output = np.zeros_like(binary_vector)
    for index in range(len(binary_vector) - (window_size + 1)):
        if binary_vector[index:index+window_size].all() == 1:
            num_behaviours += 1
            output[index:index+window_size] = np.ones(window_size)
    print('{} instances of behaviour found after filtering'.format(num_behaviours))

    return output

def annotate_video(video_file, data, data_names, binary_vector_dict):

    # assume numpy 1D data incoming 
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print('frame size: ',frame_width,'x',frame_height)
    framerate = 30
#    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES,START_POINT)

    # assert num_frames == len(dot_data_x), "video frames must be equal in length to dot frames"

    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.splitext(video_file)[0]+'_dots.mp4', fourcc, framerate, (frame_width,frame_height))    

    male_color = (255,0,0)
    female_color = (0,0,255)
    
    male_cm = iter(cm.Reds_r(np.linspace(0,1,5)))
    female_cm = iter(cm.Blues_r(np.linspace(0,1,5)))
    color_dict = {} 
    
    pos_dict = {}
    shift = 40
    
    male_pos = (20,50)
    female_pos = (int(frame_width-350),50)
    centre_pos = (int(frame_width/2),50)
    
    for b in binary_vector_dict.keys():
        if "female" in b:
            pos_dict[b] = female_pos 
            female_pos = (female_pos[0],shift+female_pos[1])

        elif "male" in b:
            pos_dict[b] = male_pos 
            male_pos = (male_pos[0],shift+male_pos[1])
        else: 
            pos_dict[b] = centre_pos
            centre_pos = (centre_pos[0],shift+centre_pos[1])
    print(pos_dict)


    for k in range(0,len(data)-1,2): 
        p = data_names[k]
        if "female" in p:
            color_dict[p] = next(female_cm)
        elif "male" in b:
            color_dict[p] = next(male_cm)
    
    
    for i in range(START_POINT,END_POINT):
        ret, frame = cap.read()
        for k in range(0,len(data)-1,2): 
            
            color = color_dict[data_names[k]]
            
#            if data_names[k][0] is 'f':
#                color = next(female_cm) #female_color
#            else:
#                color = next(male_cm) # male_color

            dot_data_x = data[k]
            dot_data_y = data[k+1]
            
            cv2.circle(frame, 
             (int(round(dot_data_x[i])),int(round(dot_data_y[i]))), 
              10, 
              color,
              -1) 
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2            

        for behaviour_type in binary_vector_dict:
            binary_vector = binary_vector_dict[behaviour_type]          

            bottomLeftCornerOfText = pos_dict[behaviour_type]
            if binary_vector[i] == 1:
                cv2.putText(frame,
                        behaviour_type,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
        out.write(frame)

    cap.release()
    out.release()



def main():
    
    data, data_names = load_data_dict()
    
    video_file = r"C:\Users\2018_Group_a\Documents\DLC-behavior-analysis\mouse_long.avi"
    
    with open("ethogram.pkl", "rb") as f:
        ethogram = pickle.load(f)
        
    dict_list_to_numpy(ethogram)
    
    for behaviour in ethogram.keys():
        ethogram[behaviour] = filter_binary(ethogram[behaviour], 13)
#    
    annotate_video(video_file, data, data_names, ethogram)
    
if __name__ == '__main__':
    main()

