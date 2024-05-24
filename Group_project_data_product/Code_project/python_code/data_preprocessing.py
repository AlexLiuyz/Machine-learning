import os
import shutil
from os.path import join as opj

def get_diagrams(input_root, output_root, models, subtype,diagram_name,topk = None):
    os.makedirs(output_root, exist_ok=True)
    model_pools = os.listdir(input_root)
    if models == None:
        models = [model for model in model_pools if not model.startswith(".") ]

    for model in models:
        select_model = None
        for m in model_pools:
            if model in m:
                select_model = m
                break
        if select_model is  None:
            continue

        if topk is not None:
            file_root = opj(input_root,select_model, subtype,f"diagrams_top_{topk}_topics")
        else:
            file_root = opj(input_root,select_model,subtype)

        for root, directories, files in os.walk(file_root):
            for f in files:
                name,base = os.path.splitext(f)
                if diagram_name == name:
                    input_path = opj(root, f)
                    # print(input_path)
                    output_path = opj(output_root,f"{select_model}.png")
                    os.makedirs(os.path.dirname(input_path), exist_ok = True)

                    shutil.copyfile(input_path,output_path)

input_root = "/Users/yuanyunchen/Desktop/test/embeddings_diagrams"
output_root = "/Users/yuanyunchen/Desktop/test/output"


# round_name = "top_5_word_scores_8"
# models = ["meta","small""google"]
models = None
# subtype = "HDBSCAN_cluster__and__umap_dr"
subtype = "k_means_8_cluster__and__LSA_dr"
# diagram_name = "top_5_word_scores"
diagram_name = "top_10_word_scores"
topk = 8

output_root = opj(output_root,  diagram_name + subtype)
get_diagrams(input_root, output_root, models, subtype,diagram_name,topk)



# import pandas as pd
# import csv
#
# #
# class_map = {
#     "Other": 0,
#     "World": 1,
#     "Sports":2,
#     "Business": 3,
#     "Sci/Tech": 4
# }
#
# df = pd.read_csv("./sample2.csv")
#
# counters = {}
# output_list = []
#
# for i in range(len(df)):
#     label = df.loc[i, "GPT_category"]
#     confidence = df.loc[i, "GPT_confidence_score"]
#     index = df.iloc[i,0]
#     data = df.loc[i, "headline_text"]
#
#     bar = 0.9
#     if confidence >= bar:
#         if label != "Other":
#             if label in counters:
#                 if counters[label] >= 244:
#                     continue
#                 counters[label] += 1
#             else:
#                 counters[label] = 1
#             row = [index, data, class_map[label]]
#             output_list.append(row)
#
# headers = ["index", "data", "label"]
# output_path = "/Users/yuanyunchen/Desktop/test/test_dataset_no_zero.csv"
# with open(output_path, "w+") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(headers)
#     writer.writerows(output_list)
#
# print(counters)
# print(sum(counters.values()))



import pandas as pd
from sklearn.metrics import classification_report

def get_classification_metrics(csv_file):
    df = pd.read_csv(csv_file)
    labels = df.label
    preds = df.pred

    return classification_report(labels, preds)

csv_file = "dataset_with_pred.csv"
metrics = get_classification_metrics(csv_file)
print(metrics)

import numpy as np

emb = np.load("/Users/yuanyunchen/Desktop/test/reduced_embeddings_tf_idf.npy")
print(emb)
import numpy as np

# topics = np.load("8_topics.npy", allow_pickle=True)
probs = np.load("8_probs.npy",allow_pickle=True)

print(probs)