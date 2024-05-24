

from sklearn.cluster import AgglomerativeClustering

DEFAULT_N_COMPONENTS = 5
DEFAULT_N_CLUSTERS = 8
MIN_CLUSTER_SIZE = 50

###

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer,TfidfTransformer

tf_idf_emb = TfidfVectorizer()
hashing_emb = HashingVectorizer()
word_count_emb = CountVectorizer()
# tf_idf_transformer_emb = TfidfTransformer()

traditional_emb_models_dict = {
    "tf_idf_emb": tf_idf_emb,
    # "hashing_emb": hashing_emb,
    "word_count_emb": word_count_emb,
}

base_sentence_transformer_emb = SentenceTransformer("all-MiniLM-L6-v2")
# sota_sentence_transformer_emb = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
small_sentence_transformer_emb =  SentenceTransformer("paraphrase-albert-small-v2")
semantic_search_emb =  SentenceTransformer("multi-qa-mpnet-base-dot-v1")
bing_search_emb =  SentenceTransformer("msmarco-bert-base-dot-v5")
# clip_emb = SentenceTransformer("clip-ViT-L-14")
google_questions_emb = SentenceTransformer("nq-distilbert-base-v1")
meta_questions_emb = SentenceTransformer("facebook-dpr-ctx_encoder-multiset-base")


deep_emb_model_dict = {
"base_sentence_transformer_emb": base_sentence_transformer_emb,
# "sota_sentence_transformer_emb": sota_sentence_transformer_emb,
"small_sentence_transformer_emb": small_sentence_transformer_emb,
"semantic_search_emb": semantic_search_emb,
"bing_search_emb": bing_search_emb,
# "clip_emb": clip_emb,
"google_questions_emb": google_questions_emb,
"meta_questions_emb": meta_questions_emb
}

embeddings_model_dict = {**traditional_emb_models_dict, **deep_emb_model_dict}


from umap import UMAP
from sklearn.decomposition import TruncatedSVD,LatentDirichletAllocation,DictionaryLearning, PCA

LDA_dr = LatentDirichletAllocation(n_components=DEFAULT_N_COMPONENTS)
LSA_dr = TruncatedSVD(n_components=DEFAULT_N_COMPONENTS)
dict_dr = DictionaryLearning(n_components = DEFAULT_N_COMPONENTS)
umap_dr = UMAP(n_neighbors=15, n_components=DEFAULT_N_COMPONENTS, min_dist=0.0, metric='cosine', random_state=42)
PCA_dr = PCA(n_components=DEFAULT_N_COMPONENTS)

dr_model_dict = {
    'LSA': LSA_dr,
    'DictionaryLearning': dict_dr,
    'umap': umap_dr,
    'PCA': PCA_dr,
    'LDA': LDA_dr,
}

###

# Improving Default Representation
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

k_means_cluster = KMeans(n_clusters=DEFAULT_N_CLUSTERS)
HDBSCAN_cluster = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
HDBSCAN_cluster_8 = HDBSCAN(min_cluster_size=8,max_cluster_size=8, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

###
# Controlling Number of Topics
from sklearn.feature_extraction.text import CountVectorizer

CountVectorizer_vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

from bertopic.representation import KeyBERTInspired

keybert_representation = KeyBERTInspired()

