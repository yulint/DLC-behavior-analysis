# annotate video 

import numpy as np 
import cv2
import pickle

# only keep long sequences of behaviours 
# window size is minimum length of behaviour
# behaviours lasting less will be filtered
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

def add_dots_to_video(video_file, dot_data_x, dot_data_y, color):

	# assume numpy 1D data incoming 
	cap = cv2.VideoCapture(filename_in)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	print('frame size: ',frame_width,'x',frame_height)
	framerate = 30
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	assert num_frames == len(dot_data), "video frames must be equal in length to dot frames"

	# fourcc = cv2.VideoWriter_fourcc(*'avc1')
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(os.path.splitext(video_file)[0]+'_dots.mp4', fourcc, framerate, (frame_width,frame_height))    

	for i in range(num_frames):
		ret, frame = cap.read()
		cv2.circle(frame, center=[dot_data_y[i],dot_data_x[i]], radius=1) 
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

    for i in range(num_frames):
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

    video_file = "../mouse_clip.mp4"
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames)
    
    with open("ethogram.pkl", "rb") as f:
        ethogram = pickle.load(f)
    
    for behaviour in ethogram.keys():
        ethogram[behaviour] = filter_binary(ethogram[behaviour], 13)
    
    annotate_video_from_dict(video_file,"../mouse_annotated.avi",ethogram, num_frames)
    
if __name__ == '__main__':
	main()

