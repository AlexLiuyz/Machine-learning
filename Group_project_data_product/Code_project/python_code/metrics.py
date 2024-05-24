import ast

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

import csv
import os
from os.path import join as opj
import pandas as pd

news_df = pd.read_csv("/root/autodl-tmp/workspace/BERTopic/data/new_text_withstem.csv")
# Extract

cleaned_text = news_df['headline_cleaned_text']
original_text = news_df['headline_text']
num_samples = 100000
cleaned_text = cleaned_text[:num_samples]
cleaned_text = [cleaned_text[i] if isinstance(cleaned_text[i],str) else original_text[i] for i in range(len(cleaned_text))]

def get_topic_metrics(cleaned_text:str,bertopic:list,topk:int,measure="c_uci"):
    bertopic_topics=[]
    for topic in bertopic:
        topics = ast.literal_eval(topic)
        bertopic_topics.append(topics)
        # print(topics)
    corpus = [text.split(" ") for text in cleaned_text]
    coherence = Coherence(texts=corpus,
                            topk=topk, measure=measure)

    diversity = TopicDiversity(topk=topk)

    # print(1)
    coherence = coherence.score({"topics": bertopic_topics})
    # print(2)
    diversity = diversity.score({"topics": bertopic_topics})
    # print(4)

    return coherence, diversity


def evaluate_group_models(input_root,subdirname, output_path,n_topics, topk = 10,measure = "c_uci"):
    dirs = os.listdir(input_root)
    output_list = []

    for directory in dirs:
        # if directory in ['umap']:
        #     continue
        if directory.startswith("."):
            continue
        
        # model_name = directory[:-4]
        model_name = directory
        csv_path = opj(opj(input_root,directory,subdirname,"topic_representations.csv"))
        if not os.path.exists(csv_path):
            # print(csv_path)
            continue
        df = pd.read_csv(csv_path)
        
        index = 0
        if df.loc[0, "Topic"] == -1:
            index = 1
        
        if n_topics > len(df):
            n_topics = len(df)
            
        bertopic = df.loc[index:index + n_topics, "Representation"].tolist()
        coherence, diversity = get_topic_metrics(cleaned_text,bertopic,topk,measure=measure)
        print(model_name, coherence,diversity)
        output_list.append([model_name,f"{coherence:.4f}",f"{diversity:.4f}"] )
    headers = ["models", "TC", "TD"]
    
    output_root = os.path.dirname(output_path)
    os.makedirs(output_root, exist_ok = True)
    with open(output_path, "w+") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(output_list)

input_root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_clustering"
csv_output_root = "/root/autodl-tmp/workspace/BERTopic/output/evaluation"
subdirname = ""

round_name = "test_clustering"
measure = "c_v"
# for n_topics in [8,50,800]:
for n_topics in [800]:
    output_path = opj(csv_output_root,f"{round_name}_{n_topics}.csv")
    evaluate_group_models(input_root,subdirname, output_path,measure = measure,n_topics = n_topics)

