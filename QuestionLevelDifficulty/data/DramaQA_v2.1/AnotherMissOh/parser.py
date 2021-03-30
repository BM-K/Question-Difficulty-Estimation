#-*- coding:utf-8 -*-
import json
import csv
import os
from collections import OrderedDict

def json_load(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)

def writer_tsv(output_file, data_list):
    with open(output_file+'.tsv', "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t')
        for data in data_list:
            writer.writerow(data)

def get_utterance(vid, script, container):
    temp_utter = []
    scenario = vid.split('_')[2]
    new_container = []
    utterance = ""
    
    if len(container) == 2:
        for i in range(container[0], container[1] + 1):
            new_container.append(i)
    
    for idx in range(len(new_container)):
        if len(str(new_container[idx])) == 1:
            new_container[idx] = '000' + str(new_container[idx])
        elif len(str(new_container[idx])) == 2:
            new_container[idx] = '00' + str(new_container[idx])
        elif len(str(new_container[idx])) == 3:
            new_container[idx] = '0' + str(new_container[idx])
        elif len(str(new_container[idx])) == 4:
            new_container[idx] = (str(new_container[idx]))
        else:
            print("==ERROR TYPE 1==")
            exit()
    
    for idx in range(len(new_container)):
        new_container[idx] = vid[:-4] + new_container[idx]
    
    for vid_name in new_container:
        try:
            value = script[vid_name]["contained_subs"]
            for i in range(len(value)):
                temp_utter.append(value[i]["utter"].strip())
        except KeyError:
            continue
    
    utter = list(OrderedDict.fromkeys(temp_utter))
    utter = ' '.join(utter)
    
    return utter

if __name__=="__main__":

    data_name_list = ["AnotherMissOhQA_train_set.json", "AnotherMissOhQA_val_set.json", "AnotherMissOhQA_test_set.json"]
    script_file = "AnotherMissOh_script.json"
    
    script_data = json_load(script_file)

    for data_name in data_name_list:
        cur_data = json_load(data_name)
        inform = []
        qa_level_set = set()
        for data in cur_data:
            qid = data["qid"]
            vid = data["vid"]
            q_level_mem = data['q_level_mem']
            q_level_logic = data['q_level_logic']
            qa_level = data['qa_level']
            que = data['que']
            answers = data['answers']
            shot_contained_number = data["shot_contained"]
            
            utterance = get_utterance(vid, script_data, shot_contained_number)

            inform.append([vid, qid, que, answers, utterance, q_level_mem, q_level_logic, qa_level, shot_contained_number])
            qa_level_set.add(q_level_mem)
            
        writer_tsv(data_name.split('_')[1], inform)


