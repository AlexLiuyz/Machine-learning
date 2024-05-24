
import pandas as pd
from bertopic import BERTopic
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import time

from umap import UMAP


news_df = pd.read_csv("/root/autodl-tmp/workspace/BERTopic/data/new_text_withstem.csv")
# Extract

cleaned_text = news_df['headline_cleaned_text']
original_text = news_df['headline_text']
cleaned_text = [cleaned_text[i] if isinstance(cleaned_text[i],str) else original_text[i] for i in range(len(cleaned_text))]

num_samples = 100000
data = cleaned_text[:num_samples]


def get_a_group_of_document_dirtribution_diagrams(input_root, output_root,data):
  os.makedirs(output_root, exist_ok = True)
  topic_model = BERTopic.load(opj(input_root, "model"))
#   topics_distr = np.load(opj(output_root, "topics_distr.npy"))
#   _, num_topics = topics_distr.shape
  embeddings = np.load(opj(input_root, "embeddings.npy"))

  reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
  print("Finish reduce embeddings.")

  visualize_samples_list = [500, 2000, 10000,20000,50000,100000]
  output_topics_list = [8,20,50]
  for visualize_samples in visualize_samples_list:
    for output_n_topics in output_topics_list:
    #   if num_topics < output_n_topics:
    #     continue

      try:
        diagram = get_document_dirtribution_diagram(visualize_samples, topic_model, output_n_topics, data,reduced_embeddings)
      except Exception as e:
        # print(e)
        continue

      output_path = opj(output_root, f"samples_{visualize_samples}_topics_{output_n_topics}.png")
      diagram.write_image(output_path)
      print(f"finish samples_{visualize_samples}_topics_{output_n_topics}")



def get_document_dirtribution_diagram(visualize_samples, topic_model,  output_n_topics, data,reduced_embeddings):
  sample_percentage = visualize_samples / len(data) if  len(data) > visualize_samples else None
  # sample_percentage = 0.2
  tsne_documents_distribution = topic_model.visualize_documents(data,
                                reduced_embeddings=reduced_embeddings,sample = sample_percentage,
                                custom_labels=True,hide_annotations=True, topics = [i+1 for i in range(output_n_topics)])
  return tsne_documents_distribution

def add_document_dirtribution_diagrams_for_all(input_root,data):
  for root, dirs, files in os.walk(input_root):
    for f in files:
      if f == "model":
        file_path = opj(root, f)
        print(f"Start {file_path} at {time.ctime(time.time())}")

        input_root = os.path.dirname(file_path)
        output_root = opj(input_root, "documents_distribusion_graphs")
        try:
          get_a_group_of_document_dirtribution_diagrams(input_root, output_root,data)
        except Exception as exception:
          print(f"Error: {exception}")

        print(f"Finish {file_path} at {time.ctime(time.time())}")

# input_root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings"
# add_document_dirtribution_diagrams_for_all(input_root,data)


def get_heat_maps(input_root, output_root,data):
    os.makedirs(output_root, exist_ok = True)
    topic_model = BERTopic.load(opj(input_root, "model"))
    # topics_distr = np.load(opj(output_root, "topics_distr.npy"))
    # _, num_topics = topics_distr.shape
    
    output_n_topics_list = [8,20,50]
    for output_n_topics in output_n_topics_list:
        # try: 
        topic_similarity_heat_map = topic_model.visualize_heatmap(width=1000, height=1000,top_n_topics = output_n_topics)
        # except:
        #     continue
        output_path = opj(output_root, f"{output_n_topics}_topics_heat_map.png")
        topic_similarity_heat_map.write_image(output_path)
input_root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings/tf_idf_emb"
output_root = "/root/autodl-tmp/workspace/BERTopic/output/hhh"
get_heat_maps(input_root, output_root,data)

def add_heat_maps_for_all(input_root,data):
  for root, dirs, files in os.walk(input_root):
    for f in files:
      if f == "model":
        file_path = opj(root, f)
        print(f"Start {file_path} at {time.ctime(time.time())}")

        input_root = os.path.dirname(file_path)
        output_root = opj(input_root, "topic_heat_maps")
        # try:
        get_heat_maps(input_root, output_root,data)
        # except Exception as exception:
        #   print(f"Error: {exception}")

        print(f"Finish {file_path} at {time.ctime(time.time())}")


# input_root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings/sota_sentence_transformer_emb/HDBSCAN_cluster__and__umap_dr"
# add_heat_maps_for_all(input_root,data)
    
    
