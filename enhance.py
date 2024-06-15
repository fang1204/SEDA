import random
import re
import pandas as pd
import json
import os, csv
import numpy as np
from operator import itemgetter
# from tqdm import tqdm

def enhance(enhance_id,config,logger, format, split_type):
    model="w2ner"
    if "cadec" in config.dataset:
        data_format = format        #dev, train/test
        file_format = 'cadec'      #share13/share14/cadec
        ans_data_path = format      #dev, train /test
        # answer_file='acl/data/cadec/adr/ann'
        # folder_path = 'acl/data/'+file_format+'/adr/text'
        # path = "acl/cadec/new_"+file_format+"/"+file_format+"_word_list.json"
        answer_file='acl/cadec/ann'
        folder_path = 'acl/'+file_format+'/text'
        path = "acl/"+file_format+"/"+file_format+"_word_list.json"

        pattern = r'[\w]+|[^\s\w]'
        pattern_s = r'[\w]+|\n|[\s]+|[^\s\w]'
    else:
        data_format = format        #dev, train/test
        if "share13" in config.dataset:
            file_format = 'share13'       #share13/share14/cadec
        else:
            file_format = 'share14'      #share13/share14/cadec

        if format=="test":
            ans_data_path=format
        else:
            ans_data_path="train"
        # answer_file='acl/data/'+file_format+'/'+ans_data_path+'.ann'
        # folder_path = 'acl/data/'+file_format+'/'+ans_data_path+'/text'
        # answer_file='acl/' + file_format + '/ann/' + ans_data_path + '.ann'
        # folder_path = 'acl/' + file_format + '/ann/' + ans_data_path + '/text'
        answer_file='acl/' + file_format + '/' + ans_data_path + '.ann'
        folder_path = 'acl/' + file_format + '/' + ans_data_path + '/text'
        if "our" in split_type:
            # our pattern 
            pattern = r'[\w]+|[^\s\w]'
            pattern_s = r'[\w]+|\n|[\s]+|[^\s\w]'
            # path = "acl/data/cadec/new_"+file_format+"/"+file_format+"_"+data_format+".json"
            # path = "acl/cadec/new_"+file_format+"/"+file_format+"_word_list.json"
            path = "acl/"+file_format+"/"+file_format+"_word_list.json"

        # if "dai" in split_type:
        #     # new pattern - dai
        #     contractions = set(["n't", "'s", "'ve", "'re", "'ll", "'d", "'m"])
        #     contractions |= set([x.replace("'", "’") for x in contractions])

        #     contractions_pattern_1 = "\w+(?="+")|\w+(?=".join(list(contractions))+")"
        #     contractions_pattern_2 = "|".join(list(contractions))

        #     pattern = "\w+(?=n't)|~\w+|"+contractions_pattern_1+"|\w+#\w*|\w+=\w*|\w+‘\w*|\w+“\w*|"+contractions_pattern_2+"|"+r"""[\w\d]+[*]+|[|]+|\w+"\w+|\w+'\w+|[\w]+|[^\s\w]"""
        #     pattern_s = "\w+(?=n't)|~\w+|"+contractions_pattern_1+"|\w+#\w*|\w+=\w*|\w+‘\w*|\w+“\w*|"+contractions_pattern_2+"|"+r"""[\w\d]+[*]+|[|]+|\w+"\w+|\w+'\w+|[\w]+|[^\s\w]"""+"|\n|[\s]+"
        #     path = "acl/data/cadec/new_"+file_format+"/"+file_format+"_"+data_format+"_new.json"
        #     # path = "answer/"+file_format+"/ann/"+file_format+"_word_list.json"
        #     # path = "acl/cadec/new_"+file_format+"/"+file_format+"_word_list.json"
            

    
    #share13
    to_enhance_count = "enhance_"+str(enhance_id)
    
    #share13/14
    ans = pd.read_csv(answer_file, names=["file_id", "label", "start_end", "entity", "other"], keep_default_na=False, na_values=['NaN', 'null'], delimiter='\t')
    ans_tmp = ans.groupby(['file_id'])
    sentence_length = []

    ans_split = pd.DataFrame()
    file_list = os.listdir("./ann/"+config.enhance_project+"/"+data_format)
    for enhance_file in file_list:
        tmp_df = pd.read_csv("./ann/"+config.enhance_project+"/"+data_format+"/"+enhance_file, names=["file_id", "label", "start_end", "entity", "other"], keep_default_na=False, na_values=['NaN', 'null'], delimiter='\t')
        tmp_df['entity_bound'] = tmp_df['start_end'].apply(lambda x: ",".join([x.split(',')[0],x.split(',')[-1]]))
        # tmp_df = tmp_df.drop_duplicates(subset=['file_id','entity_bound'])
        ans_split = pd.concat([ans_split,tmp_df])

    if len(file_list)>1:
        ans_split_df_duplicated = ans_split.duplicated(subset=['file_id','entity_bound'], keep=False)
        # ans_split = ans_split[ans_split_df_duplicated].groupby(['file_id','entity_bound']).filter(lambda x: len(x) >= (len(file_list))*0.75)
        ans_split = ans_split[ans_split_df_duplicated].groupby(['file_id','entity_bound']).filter(lambda x: len(x) == (len(file_list)))
        ans_split = ans_split.drop_duplicates(subset=['file_id','entity_bound'])

    # # ans_split = p_c.drop_duplicates(subset=['file_id','start_end'])
    ans_split_tmp = ans_split.groupby(['file_id'])

    # 設定資料夾路徑
    # 設定輸出 JSONL 檔案的路徑
    # D:\acl\data\cadec\new_share13\ann_to_share13
    main_block_size =0
    ES_action_state  = config.ES_action_state
    NES_action_state = config.NES_action_state
    is_write =       1      # 1寫入        / 0 不寫
    get_all_entity = 1     # 1取得全部實體 / 0 則否
    total_sentence = 0 
    add_train_random = 0            #train random 0/1 沒有/有
    contain_merge_conflict_span = config.contain_merge_conflict_span # 0 衝突區間個體/ 1 衝突區間全部

    '''
    action_state
    {
        0: not work
        1: work
    }
    '''
    support_strategy = config.support_strategy
    '''
    support_strategy
    {
        0: look forward
        1: look backward
        2: look both side
    }
    '''
    look_forward_step = config.look_forward_step
    look_backward_step= config.look_backward_step

    # to_enhance_count
    if main_block_size==0:
        main_block_mode = "dynamic_size"
    else:
        main_block_mode = str(main_block_size)
    # to_enhance_count
    if support_strategy==0 and (ES_action_state or NES_action_state):
        to_enhance_count += "_look_forward"
        domain_t = main_block_mode+"-"+"F"+str(look_forward_step)+"-"+"B0"
    elif support_strategy==1 and (ES_action_state or NES_action_state):
        to_enhance_count += "_look_backward"
        domain_t = main_block_mode+"-"+"F0"+"-"+"B"+str(look_backward_step)
    elif support_strategy==2 and (ES_action_state or NES_action_state):
        to_enhance_count += "_look_both_side"
        domain_t = main_block_mode+"-"+"F"+str(look_forward_step)+"-"+"B"+str(look_backward_step)
    else:
        to_enhance_count += ""
    # 7802 0  7882 1  7930 3

    if ES_action_state + NES_action_state==2:
        domain_s = "_ALL"+"-"+domain_t
    elif ES_action_state==1:
        domain_s = "_ES"+"-"+domain_t
    elif NES_action_state==1:
        domain_s = "_NES"+"-"+domain_t
    else:
        domain_s = "_None"

    if not os.path.exists("enhance_data/"+config.enhance_project):
        os.makedirs("enhance_data/"+config.enhance_project)

    output_path = "enhance_data/"+config.enhance_project+"/"+model+"_"+data_format+"_"+to_enhance_count+domain_s+".json"
    # print(output_path)

    # 
    merge_intervals_limit = 5 # char 數為單位
    entity_intervallimit = 20       # 設35會錯
    # 10*5
    total_sentence = 0
    total_entity = 0
    count = 0
    enhance_sentence_length = []
    single_entity = 0


    with open(path)as f_ori:
        dic_f_wordlist = json.load(f_ori)

    def index_label(test_list):
        pre_text = test_list[-1]
        newList = []
        textList = []
        start = 0
        for i in range(len(pre_text)):
            start =  test_list[i][0]
            p_t_s = re.findall(pattern_s,pre_text[i])
            # p_t = re.findall(pattern,pre_text[i])
            for it,t in enumerate(p_t_s):        
                end = start + len(t)
                if t != ' ':
                    textList.append(t)
                    newList.append([start, end])
                start = end
        newList.append(textList)
        return newList

    def search_entity_index(sentence, ran, d):
        list_ner = []
        # print(sentence, ran, d)
        for i in range(len(d[-2])):
            entity_start = d[i][0] - ran[0]
            entity_end = d[i][1] - ran[0]
            
            if i == 0:
                entity_ner = len(re.findall(pattern,sentence[0:entity_start]))
                list_ner.append(entity_ner)
                
            else:
                entity_ner = entity_ner + len(re.findall(pattern,sentence[entity_record:entity_start])) + 1
                if entity_ner < len(re.findall(pattern,sentence)):
                    list_ner.append(entity_ner)
            entity_record = entity_end
        
        # new_list = []
        # for i in range(list_ner[0],list_ner[-1]+1):
        #     new_list.append(int(i))
        
        # if int(enhance_id)!=config.enhance_count-1:
        #     list_ner = new_list

        
        return list_ner

    def merge_span(ner_start_end_dict):
        new_item_s_e = []
        for k, vv in ner_start_end_dict.items():
            new_item_s_e.append(vv)
        new_item_s_e = sorted(new_item_s_e,key=itemgetter(0))
        mergedData = []
        start, end = new_item_s_e[0]
        for pair in new_item_s_e[1:]:
            if pair[0] <= end:
                end = max(end, pair[1])
            else:
                mergedData.append([start, end])
                start, end = pair

        mergedData.append([start, end])
        return mergedData
    def sentence_ner(diease, sentence_dict):
        nonlocal single_entity
        ans = {}
        ans_single = {}
        for d in diease:
            for k, v in sentence_dict.items():
                sentence = v[0]
                ran = k
                if ran[0]<=d[0][0] and d[-3][-1]<=ran[1]:

                    list_ner = search_entity_index(sentence, ran, d)        
                    if ran not in ans_single:     
                        ans_single[ran] = []
                        ans_single[ran].append({"index":list_ner, "type":d[-1]})
                    else:
                        ans_single[ran].append({"index":list_ner, "type":d[-1]})    
                    single_entity+=1
                    break

        for k, v in sentence_dict.items():
            
            for d in diease:
                sentence = v[0]
                ran = k
                if ran[0]<=d[0][0] and d[-3][-1]<=ran[1]:
                    list_ner = search_entity_index(sentence, ran, d)        
                    if ran not in ans:
                        
                        ans[ran] = []
                        ans[ran].append({"index":list_ner, "type":d[-1]})
                    else:
                        ans[ran].append({"index":list_ner, "type":d[-1]})    
        # if int(enhance_id)!=config.enhance_count-1:
        #     for k, v in ans.items():
        #         if len(v) >1:
        #             merge_dict = {}
        #             merge_list = []
        #             for i in v:
        #                 if i["index"][0] not in merge_dict:
        #                     merge_dict[i["index"][0]] = [i["index"][0], i["index"][-1]]
        #                 else:
        #                     nl = max(merge_dict[i["index"][0]][-1], i["index"][-1])
        #                     merge_dict[i["index"][0]] = [i["index"][0], nl]
                
        #             mergedData = merge_span(merge_dict)
        #             merge_list = []
        #             for i in mergedData:
        #                 merge_list.append({'index': [m for m in range(int(i[0]), int(i[-1])+1)], 'type': d[-1]})
        #             ans[k] = merge_list
                    
        return ans

    def merge_intervals(intervals):
        if not intervals:
            return []

        # Sort intervals based on the start value
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]        
        for current in intervals[1:]:
            # Get the last interval in merged
            last_interval = merged[-1]

            if current[0] - last_interval[1] <= merge_intervals_limit and (last_interval[1] - last_interval[0])<entity_intervallimit:
            
                merged[-1] = [last_interval[0], max(last_interval[1], current[1])]
            else:
                # Add the current interval to merged
                merged.append(current)

        return merged

    def ans_list(ans, sentence_dict, filename):
        nonlocal enhance_sentence_length
        list_dict = []
        count = 0
        for k, value in sentence_dict.items():
            v = value[0]
            span_id = value[1]
            start=int(k[0])
            end = int(k[1])
            word_list = re.findall(pattern_s, v)
            # word_index_list = word_index(word_list)
            word_index_list = []
            for word in word_list:
                end = start+len(word)
                if word.strip()!="":
                    word_index_list.append([start, end])
                start=end
            # print(word_index_list)
            json_dict = {
                "sentence":[], "ner":[],
                "filename": filename, "word_start_end":word_index_list,
                "span_id":span_id
            }
            json_dict["sentence"] = re.findall(pattern, v)
            if k in ans:
                json_dict["ner"] = ans[k]
                count += len(ans[k])
            if json_dict["sentence"] != []:
                list_dict.append(json_dict)
                # if span_id==1:
                enhance_sentence_length.append(len(json_dict["sentence"]))
            # if len(json_dict["sentence"])>50:
            #     print(filename, json_dict["word_start_end"],json_dict["sentence"])
                # print(span_id)

        return list_dict, count

    if is_write:
        output_path = output_path
    else:
        output_path = r"pseudo.json"


    with open(output_path, 'w', encoding='utf-8') as json_file:
        all_list = []
        f_name = open("answer/"+file_format+"/id/"+data_format+".id","r").read().split("\n")
        for filename in f_name:
            # filename = "04649-004477-DISCHARGE_SUMMARY.txt"
            # filename = "00174-002042-DISCHARGE_SUMMARY.txt"
            # filename = 'LIPITOR.732'
            # file_path = folder_path+"/"+filename +".txt"
            if "cadec" in file_format:
                file_path = folder_path+"/"+filename +".txt"
            else:
                file_path = folder_path+"/"+filename
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                doc_file = f.read()
                # D:\acl\data\share2013\train\text\25844-097135-ECHO_REPORT.txt
                if main_block_mode=="dynamic_size":
                    word_number = len(re.findall(pattern,doc_file))
                    if word_number<=200:
                        main_block_size=7
                    elif 200<word_number<=350:
                        main_block_size=9
                    elif 350<word_number<=500:
                        main_block_size=11
                    elif 500<word_number<=1000:
                        main_block_size=13
                    elif 1000<word_number<=1350:
                        main_block_size=15
                    elif 1350<word_number<=1500:
                        main_block_size=16
                    elif 1500<word_number<=2000:
                        main_block_size=17
                    else:
                        main_block_size=19
                #---------------------------------
                #正確答案
                try:
                    if answer_file!="":
                        true_answers = ans_tmp.get_group(filename)
                        target_start_end = true_answers[['start_end']]
                        target_entity = true_answers[['entity']]
                        index_to_int = []
                        target_entity_list = target_entity.values.tolist()
                        target_label = true_answers[['label']].values.tolist()
                        for s_e_index, s_e in enumerate(target_start_end.values.tolist()):
                            s_e = [int(span) for span in s_e[0].split(',')]
                            index_to_int.append([s_e, target_entity_list[s_e_index][0],target_label[s_e_index][0]])
                        index_to_int = sorted(index_to_int,key=itemgetter(0))

                        items_list = []
                        item_s_e = {}
                        for item in index_to_int:
                            items = []
                            # print(item)
                            if len(item[0])>2:
                                start = 0
                                item_str = []
                                item_index = []
                                itm = item[1]
                                for index in range(0, len(item[0]), 2):     
                                    end = start + (item[0][index+1] - item[0][index])
                                    item_str.append(itm[start:end])
                                    itm = itm[end:].strip()
                                    item_index.append([item[0][index], item[0][index+1]])
                                items = item_index+[item_str]
                            else:
                                items = [item[0], [item[1]]]
                            # print(items)
                            newList = index_label(items)+[item[2]]
                            # print(newList)
                            if newList[0][0] not in item_s_e:
                                item_s_e[newList[0][0]] = [newList[0][0], newList[-2][-1]]
                            else:
                                nl = max(item_s_e[newList[0][0]][-1] ,newList[-2][-1])
                                item_s_e[newList[0][0]] = [newList[0][0], nl]
                            items_list.append(newList)
                    else:
                        items_list=[]
                except:
                    items_list=[]

                #---------------------------------
                if filename in ans_split_tmp.groups.keys():
                    split_answers = ans_split_tmp.get_group(filename)

                    split_start_end = split_answers[['start_end']]
                    split_entity = split_answers[['entity']]
                    split_index_to_int = []
                    split_entity_list = split_entity.values.tolist()
                    for s_e_index, s_e in enumerate(split_start_end.values.tolist()):
                        s_e = [int(span) for span in s_e[0].split(',')]
                        split_index_to_int.append([s_e, split_entity_list[s_e_index][0]])
                    split_index_to_int = sorted(split_index_to_int,key=itemgetter(0))
                
                    split_items_list = []
                    split_item_s_e = {}
                    for item in split_index_to_int:
                        item0 = [[item[0][index], item[0][index+1]] for index in range(0, len(item[0]), 2)]
                        item1 = item[1].split(',')
                        item0.append(item1)
                        if item0[0][0] not in split_item_s_e:
                            split_item_s_e[item0[0][0]] = [item0[0][0], item0[-2][-1]]
                        else:
                            nl = max(split_item_s_e[item0[0][0]][-1] ,item0[-2][-1])
                            split_item_s_e[item0[0][0]] = [item0[0][0], nl]
                        split_items_list.append(item0)
                    #------------------------------------------
                    new_item_s_e = []
                    for k, v in split_item_s_e.items():
                        new_item_s_e.append(v)

                    # new_item_se = merge_intervals(new_item_s_e)

                    mergedData = []
                    start, end = new_item_s_e[0]

                    for pair in new_item_s_e[1:]:
                        if pair[0] <= end:
                            end = max(end, pair[1])
                        else:
                            mergedData.append([start, end])
                            start, end = pair

                    # 加入最後一個合併的範圍
                    mergedData.append([start, end])
                    merged_data = merge_intervals(mergedData)
                    if contain_merge_conflict_span:
                        mergedDataArray= np.array(merged_data)
                    else:
                        mergedDataArray= np.array(mergedData)

                    # --------------------------
                    # pattern_s = r'[\w]+|\n|[\s]+|[^\s\w]'
                    doc_file_l = list(doc_file)
                    start = 0
                    residual_target_dict = {}
                    for value_indx, value in enumerate(merged_data):
                        end = value[0]
                        residual = "".join(doc_file_l[start:end])
                        target = "".join(doc_file_l[value[0]:value[1]])
                        residual_target_dict[2*value_indx] = residual
                        residual_target_dict[2*value_indx+1] = target
                        start = value[1]
                    residual = "".join(doc_file_l[value[1]:])
                    residual_target_dict[len(residual_target_dict)] = residual
                    
                    #-------------------------------------------------
                    tmp = []
                    finish_list = []
                    ner = []
                    total = 0
                    # total_char = 0
                    count = 0
                    ner_count = 0
                    start = 0
                    end = 0
                    pre_start_end = (start, end)
                    sentence_dict = {}
                    nest_sentence = []
                    sentence_history = []
                    for k,v in residual_target_dict.items():
                        residual_v = re.findall(pattern_s,v)
                        # print(start,end,k,residual_v)
                        if k%2!=0:
                            pre_len=total
                            pre_s = tmp
                            for r_v in residual_v:
                                if r_v.strip()!="":
                                    total += 1
                                # total_char += len(r_v)
                                count += len(r_v)
                                tmp.append(r_v)
                            
                            end = count
                            tmp_value =  "".join(tmp)                        
                            if sentence_history!=[]:
                                prev_dict = sentence_dict[(sentence_history[-1][0], sentence_history[-1][1])]

                            if total<=10 and sentence_dict!={} and len(re.findall(pattern,prev_dict[0]))<=main_block_size:
                                
                                tmp_value =  "".join(tmp)

                                prev_dict[0] = prev_dict[0]+tmp_value
                                prev_dict[1] = ES_action_state
                                prev_dict[2] = 1
                                del sentence_dict[(sentence_history[-1][0], sentence_history[-1][1])]                
                                start = sentence_history[-1][0]
                                
                                sentence_length.pop(-1)
                                total_prev_dict = len(re.findall(pattern,prev_dict[0]))

                                sentence_dict[(start, start+len(prev_dict[0]))] = prev_dict
                                sentence_length.append(len(re.findall(pattern,prev_dict[0])))
                                sentence_history.append([start, start+len(prev_dict[0])])

                                start = end
                                tmp=[]
                                total=0
                            else:
                                # print(start)
                                tmp_value =  "".join(tmp)
                                property = [tmp_value,ES_action_state,1]
                                sentence_length.append(total)
                                sentence_dict[(start, end)] = property
                                sentence_history.append([start, end])
                                tmp=[]
                                total=0
                                start = end

                        else:
                            # print(residual_v)
                            for r_v in residual_v:
                                if total >= main_block_size:

                                    end = count
                                    tmp_value =  "".join(tmp)
                                    sentence_dict[(start, end)] = [tmp_value,NES_action_state,0]
                                    sentence_history.append([start, end])
                                    sentence_length.append(total)
                                    finish_list.append(sentence_dict)
                                    start = end                                
                                    tmp=[]
                                    total=0
                                if r_v.strip()!="":
                                    total += 1
                                # total_char += len(r_v)
                                count += len(r_v)
                                tmp.append(r_v)
                    if tmp!=[] and file_format=="cadec":
                    # if tmp!=[] :
                        # print(tmp)
                        tmp_value =  "".join(tmp)
                        if tmp_value.strip()!="":
                            end = start + len(tmp_value)
                    #         # print(total)
                            sentence_length.append(total)
                            sentence_dict[(start, end)] = [tmp_value,NES_action_state,0]

                    pre = 0
                    con = 0
                    new_sentence_dict = {}
                    check_prev_sentence=""
                    new_sentence_history=[]
                    previous_sentence_state = []
                    forward_step = look_forward_step+1
                    backward_step = look_backward_step+1
                    for i, (k, v) in enumerate(sentence_dict.items()):
                        sentence_word = re.findall(pattern, v[0])
                        sentence_word_num = len(sentence_word)
                        if v[1] == 0:
                            con += sentence_word_num
                            new_sentence_dict[k] = [v[0],v[2]]
                            
                        else:
                            if data_format=="train" and add_train_random:
                                look_forward_step = random.randint(0,forward_step)
                                look_backward_step = random.randint(0,backward_step)

                            if support_strategy == 0:
                                pre = dic_f_wordlist[filename][max(0, con-look_forward_step)][0]
                                # new_sentence_dict[(pre, k[1])] = [doc_file[pre : k[1]],v[2]]
                                check_flag = np.logical_and(pre >= mergedDataArray[:, 0], pre < mergedDataArray[:, 1])
                                # if pre>4000:
                                #     print(check_flag.sum(-1))
                                if check_flag.sum(-1)==1:
                                    conflict_span = mergedDataArray[check_flag][-1]
                                    # print(conflict_span)
                                    if conflict_span[0]!=0:
                                        # if previous_sentence_state[-1]==0:
                                        pre = conflict_span[-1]
                                new_sentence_dict[(pre, k[1])] = [doc_file[pre : k[1]],v[2]]

                                con += sentence_word_num
                                new_sentence_history.append([pre,  k[1]])

                            elif support_strategy == 1:

                                con += sentence_word_num
                                pre = dic_f_wordlist[filename][:con+look_backward_step][-1][-1]
                                # if i!=0:
                                #     # k[0]=pre
                                #     new_start = new_sentence_history[-1][-1]
                                # else:
                                #     new_start = 0
                                check_flag = np.logical_and(pre >= mergedDataArray[:, 0], pre < mergedDataArray[:, 1])
                                # if check_flag.sum(-1)==1:
                                # if conflict_span[0]!=0:
                                if check_flag.sum(-1)==1:
                                    conflict_span = mergedDataArray[check_flag][0]
                                    if conflict_span[0]!=0:
                                        pre = conflict_span[0]

                                new_sentence_dict[(k[0], pre)] = [doc_file[k[0]:pre],v[2]]
                                new_sentence_history.append([k[0], pre])

                            elif support_strategy == 2:
                                
                                # if i>0 and len(sentence_dict)>1 and sentence_word_num<7:
                                #     continue
                                # try:
                                # print(look_forward_step)
                                if look_forward_step==0:
                                    pre = k[0]
                                    # print(pre)
                                else:
                                    pre = dic_f_wordlist[filename][max(0, con-look_forward_step)][0]
                                    # print(pre)

                                check_flag = np.logical_and( pre> mergedDataArray[:, 0], pre <= mergedDataArray[:, 1])
                                if check_flag.sum(-1)==1:
                                    conflict_span = mergedDataArray[check_flag][0]
                                    if conflict_span[0]!=0:
                                        
                                        # pre = conflict_span[0]
                                        if pre - conflict_span[0]<30:
                                            pre = conflict_span[0]
                                        else:
                                            pre = conflict_span[-1]
                                
                                con += sentence_word_num
                                if look_backward_step==0:
                                    pre_e = k[-1]
                                else:
                                    pre_e = dic_f_wordlist[filename][:con+look_backward_step][-1][-1]
                                    check_flag = np.logical_and(pre_e >= mergedDataArray[:, 0], pre_e < mergedDataArray[:, 1])
                                    if check_flag.sum(-1)==1:
                                        conflict_span = mergedDataArray[check_flag][-1]
                                        if conflict_span[0]!=0:
                                            
                                            # pre_e = conflict_span[-1]
                                            if conflict_span[-1] - pre_e<30:
                                                pre_e = conflict_span[-1]
                                            else:
                                                pre_e = conflict_span[0]
                                
                                
                                new_sentence_dict[(pre, pre_e)] = [doc_file[pre:pre_e],v[2]]
                                new_sentence_history.append([pre, pre_e])
                            

                        # previous_sentence_state = [sentence_word_num,v[1]]
                        
                        

                    ans = sentence_ner(items_list, new_sentence_dict)
                    list_dict, ner_count = ans_list(ans, new_sentence_dict, filename)
                    total_entity+=ner_count
                    total_sentence+=len(list_dict)
                    # if filename in ans_split_tmp.groups.keys() and filename in ans_tmp.groups.keys():
                    # #     assert ner_count==len(target_entity)
                    #     if ner_count!=len(target_entity):
                    #         print(filename)
                    all_list.extend(list_dict)
                    
                #正確答案(要放ner)/沒預測答案(全切往前看5)
                elif filename not in ans_split_tmp.groups.keys():
                    # print(filename)

                    sentence_dict = {}
                    for i in range(0,len(dic_f_wordlist[filename]),main_block_size):
                        start_end= dic_f_wordlist[filename][i:i+main_block_size]
                        if len(start_end)>5: 
                            # print(start_end)
                            sentence = doc_file[start_end[0][0]:start_end[-1][-1]]
                            sentence_dict[(start_end[0][0], start_end[-1][-1])] = [sentence,0]
                    #--------------------------------------------------------------
                    total_sentence+=len(sentence_dict)
                    ans = sentence_ner(items_list, sentence_dict)
                    list_dict, ner_count = ans_list(ans, sentence_dict, filename)
                    total_entity+=ner_count
                    # print(ner_count)
                    # if filename in ans_tmp.groups.keys():
                    #     assert ner_count==len(target_entity)
                    all_list.extend(list_dict)
            
        json.dump(all_list, json_file, ensure_ascii=False)
    # import statistics
    # total_sentence,total_entity,max(enhance_sentence_length),min(enhance_sentence_length),statistics.mean(enhance_sentence_length),enhance_sentence_length
    logger.info("Enhance {} {}: total sentence: {:3.0f} total entity: {:3.0f} single entity: {:3.0f}  longest sentence length:{:3.0f}".format(enhance_id,data_format,total_sentence,total_entity,single_entity,max(enhance_sentence_length)))
    return output_path