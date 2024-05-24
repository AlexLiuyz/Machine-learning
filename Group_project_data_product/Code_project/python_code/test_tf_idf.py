import pandas as pd
from bertopic import BERTopic
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import time

from interface import *

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_emb = TfidfVectorizer()

from umap import UMAP
from sklearn.decomposition import  TruncatedSVD

DEFAULT_N_COMPONENTS = 20

LSA_dr = TruncatedSVD(n_components=DEFAULT_N_COMPONENTS)
umap_dr = UMAP(n_neighbors=15, n_components=DEFAULT_N_COMPONENTS, min_dist=0.0, metric='cosine', random_state=42)

from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

k_means_cluster = KMeans(n_clusters=5)
HDBSCAN_cluster = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)


from bertopic.representation import KeyBERTInspired

keybert_representation = KeyBERTInspired()

from sklearn.feature_extraction.text import CountVectorizer

CountVectorizer_vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))


representation_model = {
    "KeyBERT": keybert_representation
}

# two setting and run.

# -------------------------------------------------------------------------------------------------------------------


news_df = pd.read_csv("/root/autodl-tmp/workspace/BERTopic/data/new_text_withstem.csv")
# Extract

cleaned_text = news_df['headline_cleaned_text']
original_text = news_df['headline_text']
cleaned_text = [cleaned_text[i] if isinstance(cleaned_text[i],str) else original_text[i] for i in range(len(cleaned_text))]

num_samples = 100000
data = cleaned_text[:num_samples]

# -------------------------------------------------------------------------------------------------------------------

input_root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings/tf_idf_emb/"
subtype = "HDBSCAN_cluster__and__umap_dr"

embeddings = np.load(opj(input_root, subtype, "embeddings.npy"), allow_pickle=True)
reduced_embeddings = np.load(opj(input_root, subtype, "reduced_embeddings.npy"), allow_pickle=True)

output_root = f"/root/autodl-tmp/workspace/BERTopic/output/tf_idf/first_round/{subtype}"
pipeline_single_setting(data, output_root, embedding_model = tf_idf_emb ,DR_model = LSA_dr,
                        clustering_model = k_means_cluster ,vectorizer_model = CountVectorizer_vectorizer,
                        representation_model = representation_model, 
                        output_n_topics_list = (8,20,50), embeddings = embeddings,reduced_embeddings = reduced_embeddings)

# pipeline_single_setting(data, output_root, embedding_model = tf_idf_emb ,DR_model = umap_dr,
#                         clustering_model = HDBSCAN_cluster ,vectorizer_model = CountVectorizer_vectorizer,
#                         representation_model, 
#                         output_n_topics_list = (8,20,50), embeddings = embeddings,reduced_embeddings = reduced_embeddings):