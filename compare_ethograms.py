import pandas as pd
import pickle 
import cv2
import numpy as np
import matplotlib.pyplot as plt

from video_annotate import filter_binary, smooth_binary

df = pd.read_csv("Subj_DLC_manual_annotation_SVC_AutoSave.csv")
START_FRAME = list(df["frame_start"])[0]
STOP_FRAME = list(df["frame_stop"])[-1]
with open("ethogram.pkl", "rb") as f:
	ethogram = pickle.load(f)

def label_video(video_file, binary_vector_dict):
    # assume numpy 1D data incoming 
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print('frame size: ',frame_width,'x',frame_height)
    framerate = 30
    cap.set(cv2.CAP_PROP_POS_FRAMES,START_POINT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.splitext(video_file)[0]+'_labels.mp4', fourcc, framerate, (frame_width,frame_height))  

    for i in range(START_FRAME,STOP_FRAME):
        ret, frame = cap.read()

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

fill_max = 60
cut_max = 30
for b in ethogram.keys():
	ethogram[b] = filter_binary(smooth_binary(ethogram[b],fill_max),cut_max)

ethogram['attempt_mounting'] = ethogram['grappel']
del ethogram['grappel']

unique_behaviours = list(set(df["Behaviour"]))
behaviour_list = list(df["Behaviour"])
start_stop = list(zip(df["frame_start"],df["frame_stop"]))

start_stop_dict = {}
for b in unique_behaviours:
	start_stop_dict[b] = []

count = 1
for i in range(len(behaviour_list)):
	start_stop_dict[behaviour_list[i]].append(list(start_stop[i]))
	count += 1

# fix keys
temp = []
for b in unique_behaviours:
	new_b = b.lower().replace('annogenital','anogenital').replace('.','_').replace('\t','')
	start_stop_dict[new_b] = start_stop_dict[b]
	del start_stop_dict[b]
	temp.append(new_b)
unique_behaviours = temp[:]
# male sniffing fix 
unique_behaviours.append("male_body_sniffing")
unique_behaviours.append("male_anogenital_sniffing")
start_stop_dict["male_body_sniffing"] = start_stop_dict["male_sniffing"]
start_stop_dict["male_anogenital_sniffing"] = start_stop_dict["male_sniffing"]

# make combined behaviour ethogram
# index ethogram, combine into one 
hand_ethogram = [0 for e in ethogram["sexual_pursuit"]]
for b in start_stop_dict.keys():
	for t in start_stop_dict[b]:
		hand_ethogram[t[0]:t[1]] = [0.5 for i in range(t[1]-t[0])]

output = [0 for e in ethogram["sexual_pursuit"]]
for b in unique_behaviours:
	if b == "male_sniffing":
		pass
	elif b == "mutual_anogenital_sniffing":
		for t in start_stop_dict[b]:
			output[t[0]:t[1]] = ethogram["mutual_sniffing"][t[0]:t[1]]
	else:
		for t in start_stop_dict[b]:
			output[t[0]:t[1]] = ethogram[b][t[0]:t[1]]

output[START_FRAME-5:STOP_FRAME+5]
hand_ethogram[START_FRAME-5:STOP_FRAME+5]

# percent behaving hand annotated
print(hand_ethogram[START_FRAME-5:STOP_FRAME+5].count(0.5)/len(hand_ethogram[START_FRAME-5:STOP_FRAME+5]))

# % underreported: (number of hand annotated behavior ONS - number of overlap ONS) / number hand annotated ONS
p_overlap = 1 - (hand_ethogram[START_FRAME-5:STOP_FRAME+5].count(0.5) - output[START_FRAME-5:STOP_FRAME+5].count(1)) / hand_ethogram[START_FRAME-5:STOP_FRAME+5].count(0.5)

print('percent overlap: ',p_overlap)

fig, ax1 = plt.subplots(figsize=(15,4))
ax1.plot(output[START_FRAME-5:STOP_FRAME+5], label="Automated Overlap")
ax1.plot(hand_ethogram[START_FRAME-5:STOP_FRAME+5], label = "Hand Annotation")
ax1.set_title("Comparison Ethogram")
plt.legend()
plt.show()

# make hand annotated video 
# label_video('mouse.mp4', hand_ethogram)
