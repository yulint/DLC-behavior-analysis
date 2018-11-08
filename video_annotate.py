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
	framerate = 30
	fourcc = cv2.VideoWriter_fourcc(*"XVID")
	out = cv2.VideoWriter(filename_out, fourcc, framerate, (frame_width,frame_height))    

	for i in range(num_frames):
		ret, frame = cap.read()
		font = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (20,20)
		fontScale = 1
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

def annotate_video_fromdict(filename_in, filename_out, binary_vector_dict, num_frames):
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
		# # test filter
	# binary_vector = np.zeros(10)
	# binary_vector[2:5] = [1,1,1]
	# print('raw vector: ',binary_vector)
	# filtered = filter_binary(binary_vector, 4)
	# print('filtered vector: ',filtered)

	# get video
    video_file = "../mouse_clip.mp4"
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(num_frames)
    
    with open("ethogram.pkl", "rb") as f:
        ethogram = pickle.load(f)
    
	# filter out lonely 1's 	
#	prob_0 = .1
#	binary_vector = np.random.choice([0, 1], size=(num_frames,), p=[prob_0, 1-prob_0])
#    binary_vector = filter_binary(binary_vector, 13)
#	behaviour_type = 'attacking'
#
#	annotate_video(video_file,"../mouse_annotated.avi",binary_vector,behaviour_type, num_frames)
#	
    for behav in ethogram:
        ethogram[behav] = filter_binary(ethogram[behav], 13)
    
    annotate_video_fromdict(video_file,"../mouse_annotated.avi",ethogram, num_frames)
    
if __name__ == '__main__':
	main()

