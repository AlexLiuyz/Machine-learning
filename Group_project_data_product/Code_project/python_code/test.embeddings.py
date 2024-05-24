import pandas as pd
from bertopic import BERTopic
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import time

from interface import *
from model_pools import *



news_df = pd.read_csv("/root/autodl-tmp/workspace/BERTopic/data/new_text_withstem.csv")
# Extract

cleaned_text = news_df['headline_cleaned_text']
original_text = news_df['headline_text']
cleaned_text = [cleaned_text[i] if isinstance(cleaned_text[i],str) else original_text[i] for i in range(len(cleaned_text))]

num_samples = 100000
data = cleaned_text[:num_samples]


# -------------------------------------------------------------------------------------------------------------------

root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings"
base_cluster_models = {"k_means_50_cluster": KMeans(n_clusters=50), }
base_dr_models = {"LSA_dr":LSA_dr}
vectorizer_model = CountVectorizer_vectorizer

representation_model = {
    "KeyBERT": keybert_representation,
}


for e,embedding_model in deep_emb_model_dict.items():
    print(f"# --------◅▯◊║◊▯▻   Start {e} at {time.ctime(time.time())}    ◅▯◊║◊▯▻-------")
    try:
        embeddings = embedding_model.encode(data)
    except Exception:
        embeddings = embedding_model.fit_transform(data)

    for m,clustering_model in base_cluster_models.items():
        for dr,DR_model in base_dr_models.items():
            try:
                output_root = opj(root, e,f"{m}__and__{dr}")
                print(f"# ◅▯◊║◊▯▻   Start {m}__and__{dr} at {time.ctime(time.time())}    ◅▯◊║◊▯▻")
                pipeline_single_setting(data, output_root, embedding_model,DR_model,clustering_model,vectorizer_model,representation_model, embeddings = embeddings)

                print(f" ◅▯◊║◊▯▻   finish {m}__and__{dr} at {time.ctime(time.time())}    ◅▯◊║◊▯▻")
            except Exception as exception:
                print(f"# ◅▯◊║◊▯▻  {m}__and__{dr}  Error: {exception}    ◅▯◊║◊▯▻")


