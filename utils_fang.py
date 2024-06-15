import logging
import pickle
import time
from collections import defaultdict, deque
import data_loader
import pandas as pd
import numpy as np
from operator import itemgetter

answer_miss = 0
error_predict = 0
def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)

def decode_general(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur+1):
                # THW
                if instance[cur, pre] > 1: 
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head,cur)].add(cur)
                    # post nodes
                    for head,tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head,tail)].add(cur)
            # entity
            for tail,type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])
        
        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities

def decode(outputs, entities, length, entity_compare):
    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    with open(entity_compare,"a") as f_w:
        for instance, ent_set, l in zip(outputs, entities, length):
            predicts = []
            nodes = [Node() for _ in range(l)]
            for cur in reversed(range(l)):
                heads = []
                for pre in range(cur+1):
                    # THW
                    if instance[cur, pre] > 1: 
                        nodes[pre].THW.append((cur, instance[cur, pre]))
                        heads.append(pre)
                    # NNW
                    if pre < cur and instance[pre, cur] == 1:
                        # cur node
                        for head in heads:
                            nodes[pre].NNW[(head,cur)].add(cur)
                        # post nodes
                        for head,tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                nodes[pre].NNW[(head,tail)].add(cur)
                # entity
                for tail,type_id in nodes[cur].THW:
                    if cur == tail:
                        predicts.append(([cur], type_id))
                        continue
                    q.clear()
                    q.append([cur])
                    while len(q) > 0:
                        chains = q.pop()
                        for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                            if idx == tail:
                                predicts.append((chains + [idx], type_id))
                            else:
                                q.append(chains + [idx])
            
            predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
            decode_entities.append([convert_text_to_index(x) for x in predicts])
            f_w.write("Predict: "+str(predicts)+"\n")
            f_w.write("Target:  "+str(ent_set)+"\n")
            f_w.write("Different:  "+str(ent_set - predicts)+"\n\n")
            ent_r += len(ent_set)
            ent_p += len(predicts)
            ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities


def decode_pre(outputs, entities, length, sentence_batch,enhance_id,enhance_count):
    global answer_miss,error_predict
    predict_dataframe = pd.DataFrame(columns=["file_id", "label", "start_end", "entity", "other"])
    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    find_entity_ratio=[]
    find_area_ratio=[]
    
    q = deque()
    for instance, ent_set, l, sentence_item in zip(outputs, entities, length, sentence_batch):
        predicts = []
        label_count = 0
        # all_two = 0
        label_list = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur+1):
                # THW
                if instance[cur, pre] > 1: 
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head,cur)].add(cur)
                    # post nodes
                    for head,tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head,tail)].add(cur)
                
                if instance[cur, pre]>=2 and cur >= pre:
                    label_count += 1
                    label_list.append((list(range(pre,cur+1)), instance[cur, pre]))

            # entity
            for tail,type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])
        
        
        # -------------#
        if int(enhance_id)==enhance_count-1:
            if predicts!=[]:
                p_ner_start_end_dict ={}
                for p_item in predicts:  

                    if p_item[0][0] not in p_ner_start_end_dict:
                        p_ner_start_end_dict[p_item[0][0]] = [p_item[0][0], p_item[0][-1]]
                    else:
                        p_ner_start_end_dict[p_item[0][0]] = [p_item[0][0],max(p_ner_start_end_dict[p_item[0][0]][-1],p_item[0][-1])]
                    predict_index = [sentence_item['word_start_end'][ind] for ind in p_item[0]]
                    predict_index = ",".join(str(j) for i in predict_index for j in i)
                    # out_f.write(str(predict_index)+"\t")                           #索引
                    predict_word = [sentence_item['sentence'][ind] for ind in p_item[0]]
                    predict_word = ",".join(str(i) for i in predict_word)
                    lable = data_loader.id_label(p_item[1])
                    data_to_append = pd.DataFrame([{
                        "file_id": sentence_item['filename'], 
                        "label": lable, 
                        "start_end": predict_index,
                        "entity": predict_word, 
                        "other": ""
                    }])
                    predict_dataframe = predict_dataframe.append(data_to_append)
                p_merge = merge_span(p_ner_start_end_dict)

        else:
            if predicts!=[] or label_list!=[]:
                if label_list!=[]:
                    predicts = label_list
                # -------------#
                p_ner_start_end_dict ={}
                for p_item in predicts:  

                    if p_item[0][0] not in p_ner_start_end_dict:
                        p_ner_start_end_dict[p_item[0][0]] = [p_item[0][0], p_item[0][-1]]
                    else:
                        p_ner_start_end_dict[p_item[0][0]] = [p_item[0][0],max(p_ner_start_end_dict[p_item[0][0]][-1],p_item[0][-1])]
                    predict_index = [sentence_item['word_start_end'][ind] for ind in p_item[0]]
                    predict_index = ",".join(str(j) for i in predict_index for j in i)
                    # out_f.write(str(predict_index)+"\t")                           #索引
                    predict_word = [sentence_item['sentence'][ind] for ind in p_item[0]]
                    predict_word = ",".join(str(i) for i in predict_word)
                    lable = data_loader.id_label(p_item[1])
                    data_to_append = pd.DataFrame([{
                        "file_id": sentence_item['filename'], 
                        "label": lable, 
                        "start_end": predict_index,
                        "entity": predict_word, 
                        "other": ""
                    }])
                    predict_dataframe = predict_dataframe.append(data_to_append)

                p_merge = merge_span(p_ner_start_end_dict)

        if sentence_item['ner']!=[] and predicts!=[]:
            g_ner_start_end_dict={}
            for ner in sentence_item['ner']:
                if ner['index'][0] not in g_ner_start_end_dict:
                    g_ner_start_end_dict[ner['index'][0]] = [ner['index'][0], ner['index'][-1]]
                else:
                    g_ner_start_end_dict[ner['index'][0]] = [ner['index'][0],max(g_ner_start_end_dict[ner['index'][0]][-1],ner['index'][-1])]
            g_merge = merge_span(g_ner_start_end_dict)
            # g_list = list(range(g_merge[0][0],g_merge[-1][-1]))

            find = [np.isin(row[1], p_merge[:,1]).all() for row in g_merge]
            find_entity_ratio.append(sum(find)/p_merge.shape[0])

            g_list= []
            p_list= []
            for index, row in enumerate(g_merge):
                g_list.extend(list(range(row[0],row[-1]+1)))
            for index, row in enumerate(p_merge):
                p_list.extend(list(range(row[0],row[-1]+1)))
            find_area = np.isin(np.array(g_list), np.array(p_list))
            find_area_ratio.append(find_area.sum(-1)/len(p_list))
        elif sentence_item['ner']!=[] and predicts==[]:
            find_entity_ratio.append(0.0)
            find_area_ratio.append(0.0)

    return predict_dataframe,find_entity_ratio,find_area_ratio

def merge_span(ner_start_end_dict):
    new_item_s_e = []
    for k, v in ner_start_end_dict.items():
        new_item_s_e.append(v)
    new_item_s_e = sorted(new_item_s_e,key=itemgetter(0))
    #[12,20],[13,19],[14,21] => [12,21]
    mergedData = []
    start, end = new_item_s_e[0]
    for pair in new_item_s_e[1:]:
        if pair[0] <= end:
            end = max(end, pair[1])
        else:
            mergedData.append([start, end])
            start, end = pair

    mergedData.append([start, end])
    mergedDataArray= np.array(mergedData)
    return mergedDataArray

def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r
