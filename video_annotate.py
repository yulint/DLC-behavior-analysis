# annotate video 

import numpy as np 
import cv2
import os 
import pickle

# only keep long sequences of behaviours 
# window size is minimum length of behaviour
# behaviours lasting less will be filtered

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

def add_dots_to_video(video_file, data, data_names):

	# assume numpy 1D data incoming 
	cap = cv2.VideoCapture(video_file)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	print('frame size: ',frame_width,'x',frame_height)
	framerate = 30
	#num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# assert num_frames == len(dot_data_x), "video frames must be equal in length to dot frames"

	# fourcc = cv2.VideoWriter_fourcc(*'avc1')
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(os.path.splitext(video_file)[0]+'_dots.mp4', fourcc, framerate, (frame_width,frame_height))    

	male_color = (255,0,0)
	female_color = (0,0,255)

	for i in range(len(data[0])):
		ret, frame = cap.read()

		for k in range(1,2,len(data)-1): 

			if data_names[k][0] is 'f':
				color = female_color
			else:
				color = male_color

			dot_data_x = data[k]
			dot_data_y = data[k+1]
            
			cv2.circle(frame, 
              center=(int(round(dot_data_y[i])),int(round(dot_data_x[i]))), 
              radius=1, 
              color=color) 
			out.write(frame)

	cap.release()
	out.release()

# add words to 
def annotate_video(filename_in, filename_out, binary_vector, behaviour_type, num_frames):
	'''
	filname:: string
	binary:: 1D numpy array 
	behaviour_type:: string 
	'''
	cap = cv2.VideoCapture(filename_in)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	print('frame size: ',frame_width,'x',frame_height)
	framerate = 30
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# fourcc = cv2.VideoWriter_fourcc(*'avc1')
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(filename_out, fourcc, framerate, (frame_width,frame_height))    

	for i in range(num_frames):
		ret, frame = cap.read()
		font = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (20,20)
		fontScale = 24
		fontColor = (255,255,255)
		lineType = 2

		if binary_vector[i] is 1:
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
    
def read_video(video_file):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    if ret is not True:
        print('not read')

def annotate_video_from_dict(filename_in, filename_out, binary_vector_dict, num_frames):
    '''
    filname:: string
    binary:: 1D numpy array 
    '''
    cap = cv2.VideoCapture(filename_in)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    framerate = 30
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filename_out, fourcc, framerate, (frame_width,frame_height))    

    for i in range(len(binary_vector_dict.values[0])):
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        #bottomLeftCornerOfText = (20,20)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2

        for behaviour_type in binary_vector_dict:
            binary_vector = binary_vector_dict[behaviour_type]
            
            male_counter = 0 
            female_counter = 0
            both_counter = 0
            
            if "male" in behaviour_type:
                male_counter +=1
                bottomLeftCornerOfText = (20,20) + male_counter * (0,10)
            elif "female" in behaviour_type:
                female_counter +=1
                bottomLeftCornerOfText = (frame_width-50,20) + female_counter * (0,10)
            else:
                both_counter +=1
                bottomLeftCornerOfText = (frame_width/2,20) + both_counter * (0,10)
                
            if binary_vector[i] is 1:
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
    
    video_file = "mouse_raw.mp4"
    
    read_video(video_file)
    
#    with open("ethogram.pkl", "rb") as f:
#        ethogram = pickle.load(f)
#        
#    dict_list_to_numpy(ethogram)
#    
#    for behaviour in ethogram.keys():
#        ethogram[behaviour] = filter_binary(ethogram[behaviour], 13)
#    
#    add_dots_to_video(video_file, data, data_names)
    
if __name__ == '__main__':
	main()

