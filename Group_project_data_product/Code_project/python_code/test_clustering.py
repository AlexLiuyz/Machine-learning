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

root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_clustering"
# base_cluster_models = {"k_means_50_cluster": KMeans(n_clusters=50) }
# base_cluster_models = {"HDBSCAN_cluster": HDBSCAN_cluster, "k_means_50_cluster": KMeans(n_clusters=50) }
base_cluster_models = {"k_means_8_cluster": KMeans(n_clusters=8),"k_means_50_cluster": KMeans(n_clusters=50), "k_means_800_cluster": KMeans(n_clusters=800)}
embedding_model = small_sentence_transformer_emb
vectorizer_model = CountVectorizer_vectorizer

representation_model = {
    "KeyBERT": keybert_representation
}

DR_model = UMAP(n_neighbors=15, n_components=DEFAULT_N_COMPONENTS, min_dist=0.0, metric='cosine', random_state=42)
embeddings = np.load(opj("/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings/small_sentence_transformer_emb/HDBSCAN_cluster__and__umap_dr/embeddings.npy"))
# reduced_embeddings = np.load(opj(input_root,dr, "reduced_embeddings.npy"))

for m,clustering_model in base_cluster_models.items():
    try:
        output_root = opj(root, f"{m}")
        print(f"# ◅▯◊║◊▯▻   Start {m} at {time.ctime(time.time())}    ◅▯◊║◊▯▻")
        pipeline_single_setting(data, output_root, embedding_model,DR_model,clustering_model,vectorizer_model,representation_model, 
                                embeddings = embeddings)

        print(f" ◅▯◊║◊▯▻   finish {m} at {time.ctime(time.time())}    ◅▯◊║◊▯▻")
        
    except Exception as exception:
        print(f"# ◅▯◊║◊▯▻  {m}  Error: {exception}    ◅▯◊║◊▯▻")
