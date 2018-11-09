import pandas as pd 
import pickle



manual_annotate = pd.read_csv('Subj_DLC_manual_annotation_SVC_AutoSave.csv', header = 0)

behav_tag = manual_annotate["behav"]
behav_start = manual_annotate["start"]
behav_stop = manual_annotate["stop"]

num_rows = behav_stop[len(behav_stop)-1] - behav_start[0]

behav_list = [0 for i in range(num_rows)]
behav_idx = [0 for i in range(num_rows)]

first_idx = 0

for i in range(len(behav_tag)):
    behaviour = behav_tag[i]
    start = behav_start[i]
    stop = behav_stop[i]
    num_idx = stop-start

    behav_list[first_idx:first_idx+num_idx+1] = behaviour
    behav_idx[first_idx:first_idx+num_idx+1] = range(start, stop+1, 1)
    first_idx +=num_idx

manual_behaviour_list = [behav_list, behav_idx]
print(len(manual_behaviour_list), len(manual_behaviour_list[0])) 

with open('manual_behaviour_list.pkl', 'wb') as fp:
    pickle.dump(manual_behaviour_list, fp)

with open('manual_behaviour_list.pkl', 'rb') as fp:
    test = pickle.load(fp)
    
print(len(test), len(test[0])) 



    
    

