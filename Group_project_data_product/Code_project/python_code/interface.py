import pandas as pd
from bertopic import BERTopic
import os
from os.path import join as opj
import numpy as np
import pandas as pd

from umap import UMAP

def pipeline_single_setting(data, output_root, embedding_model,DR_model,clustering_model,vectorizer_model,representation_model,output_n_topics_list = (8,20,50), embeddings = None,reduced_embeddings = None):
  # if embeddings is None:
  #   try:
  #     embeddings = embedding_model.encode(data, show_progress_bar=True)
  #   except Exception:
  #     embeddings = embedding_model.fit_transform(data)
  if embeddings is not None:
    if reduced_embeddings is None:
      try:
        reduced_embeddings = DR_model.fit_transform(embeddings)
      except:
        print(1)
        transformed_embeddings = np.abs(embeddings)
        print(2)
        reduced_embeddings = DR_model.fit_transform(transformed_embeddings)
        print(3)

  topic_model = BERTopic(
  # Pipeline models
  embedding_model=embedding_model,
  umap_model=DR_model,
  hdbscan_model=clustering_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=30,
  # nr_topics = nr_topics,
  verbose=True
)
  topics, probs = topic_model.fit_transform(data, embeddings)
  topics_distr, _ = topic_model.approximate_distribution(data, window=8, stride=4)
  topic_representation_df = topic_model.get_topic_info()


  os.makedirs(output_root, exist_ok = True)
  embeddings_path = opj(output_root, "embeddings.npy")
  np.save(embeddings_path, embeddings)
  reduced_embeddings_path = opj(output_root, "reduced_embeddings.npy")
  np.save(reduced_embeddings_path, reduced_embeddings_path)
  topic_representation_path = opj(output_root, "topic_representations.csv")
  topic_representation_df.to_csv(topic_representation_path, index = False)
  model_path = opj(output_root, "model")
  topic_model.save(model_path)
  topics_path = opj(output_root, "topics.npy")
  np.save(topics_path, topics)
  probs_path = opj(output_root, "probs.npy")
  np.save(probs_path, probs)
  topics_distr_path = opj(output_root, "topics_distr.npy")
  np.save(topics_distr_path, topics_distr)

  _, num_topics = topics_distr.shape
  for output_n_topics in output_n_topics_list:
    if output_n_topics > num_topics:
      continue

    diagrams_output_root = opj(output_root, f"diagrams_top_{output_n_topics}_topics")
    os.makedirs(diagrams_output_root, exist_ok = True)
    diagram_map = get_topic_model_diagrams(topic_model,num_topics, embeddings,data,topics, output_n_topics = output_n_topics)
    for name, diagram in diagram_map.items():
      path = opj(diagrams_output_root, f"{name}.png")
      diagram.write_image(path)


# def get_topic_model_diagrams(topic_model, num_topics,reduced_embeddings,data, topics, output_n_topics):
#   output = {}
#   # MAX_TOPICS = 20
#   # output_n_topics = MAX_TOPICS if num_topics > MAX_TOPICS else num_topics

#   topic_distance_map = topic_model.visualize_topics(top_n_topics=output_n_topics)

#   try:
#     topic_similarity_heat_map = topic_model.visualize_heatmap(width=1000, height=1000,top_n_topics = output_n_topics)
#     # topic_8_similarity_heat_map = topic_model.visualize_heatmap(width=1000, height=1000,top_n_topics = 8)
#   except Exception:
#     topic_similarity_heat_map = None
    

#   try:
#     time_chart = topic_model.visualize_topics_over_time(topics_over_time = topics, topics = [i+1 for i in range(output_n_topics)])
#   except Exception:
#     time_chart = None

#   # visualize_samples_list = [10000,50000,100000]
#   visualize_reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(reduced_embeddings)
#   # for visualize_samples in visualize_samples_list:
#   sample_percentage1 = 10000 / len(data) if  len(data) > visualize_samples else None
#   sample_percentage2 = 10000 / len(data) if  len(data) > visualize_samples else None
#   # sample_percentage = 0.2
#   try:
#     tsne_10000_documents_distribution = topic_model.visualize_documents(data,
#                                   reduced_embeddings=visualize_reduced_embeddings,sample = sample_percentage1,
#                                   custom_labels=True,hide_annotations=True, topics = [i+1 for i in range(output_n_topics)])
#     tsne_100000_documents_distribution = topic_model.visualize_documents(data,
#                                   reduced_embeddings=visualize_reduced_embeddings,sample = sample_percentage2,
#                                   custom_labels=True,hide_annotations=True, topics = [i+1 for i in range(output_n_topics)])
#   except Exception as e:
#     tsne_10000_documents_distribution = tsne_100000_documents_distribution = 0
#     print(e)
#     pass
#   # data_map_documents_distribution = topic_model.visualize_document_datamap(data, reduced_embeddings=reduced_embeddings,
#   #                                                   custom_labels=True, topics = [i+1 for i in range(output_n_topics)])
#   try:
#     top_5_word_scores = topic_model.visualize_barchart(top_n_topics = output_n_topics, n_words = 5,height = 300, width = 300) ##
#     top_10_word_scores = topic_model.visualize_barchart(top_n_topics = output_n_topics, n_words = 10,height = 300, width = 300) ##
#   except Exception as e:
#     print(e)
#     pass

#   try:
#     term_rank = topic_model.visualize_term_rank(topics = [i+1 for i in range(output_n_topics)])
#   except Exception as e:
#     print(e)
#     pass

#   try:
#     topic_hierarchy = topic_model.visualize_hierarchy(top_n_topics=output_n_topics)
#   except Exception:
#     topic_hierarchy = None


#   # diagrams = ["topic_distance_map", "topic_similarity_heat_map", "time_chart", "tsne_documents_distribution","data_map_documents_distribution",
#   #             "top_5_word_scores", "top_10_word_scores","term_rank", "topic_hierarchy"]
#   diagrams = ["topic_distance_map", "topic_similarity_heat_map", "time_chart", "tsne_10000_documents_distribution","tsne_100000_documents_distribution","data_map_documents_distribution",
#               "top_5_word_scores", "top_10_word_scores","term_rank", "topic_hierarchy"]
#   for name in diagrams:
#     if name in locals():
#       diagram = locals()[name]
#       if diagram is not None:
#         output[name] = diagram

#   return output

def get_topic_model_diagrams(topic_model, num_topics,embeddings,data, topics, output_n_topics):
  output = {}
  # MAX_TOPICS = 20
  # output_n_topics = MAX_TOPICS if num_topics > MAX_TOPICS else num_topics

  topic_distance_map = topic_model.visualize_topics(top_n_topics=output_n_topics)

  try:
    topic_similarity_heat_map = topic_model.visualize_heatmap(width=1000, height=1000,top_n_topics = output_n_topics)
  except Exception:
    topic_similarity_heat_map = None

  try:
    time_chart = topic_model.visualize_topics_over_time(topics_over_time = topics, topics = [i+1 for i in range(output_n_topics)])
  except Exception:
    time_chart = None

  visualize_samples = 100000
  sample_percentage = visualize_samples / len(data) if  len(data) > visualize_samples else None
  # sample_percentage = 0.2
  try:
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    tsne_documents_distribution = topic_model.visualize_documents(data,
                                  reduced_embeddings=reduced_embeddings,sample = sample_percentage,
                                  custom_labels=True,hide_annotations=True, topics = [i+1 for i in range(output_n_topics)])
    tsne_10000_documents_distribution = topic_model.visualize_documents(data,
                                  reduced_embeddings=reduced_embeddings,sample = 0.1,
                                  custom_labels=True,hide_annotations=True, topics = [i+1 for i in range(output_n_topics)])
    
  except Exception as e:
    tsne_documents_distribution = None
    tsne_10000_documents_distribution = None
    print(e)
    pass
  # data_map_documents_distribution = topic_model.visualize_document_datamap(data, reduced_embeddings=reduced_embeddings,
  #                                                   custom_labels=True, topics = [i+1 for i in range(output_n_topics)])
  try:
    top_5_word_scores = topic_model.visualize_barchart(top_n_topics = output_n_topics, n_words = 5,height = 300, width = 300) ##
    top_10_word_scores = topic_model.visualize_barchart(top_n_topics = output_n_topics, n_words = 10,height = 300, width = 300) ##
  except Exception as e:
    print(e)
    pass

  try:
    term_rank = topic_model.visualize_term_rank(topics = [i+1 for i in range(output_n_topics)])
  except Exception as e:
    print(e)
    pass

  try:
    topic_hierarchy = topic_model.visualize_hierarchy(top_n_topics=output_n_topics)
  except Exception:
    topic_hierarchy = None


  # diagrams = ["topic_distance_map", "topic_similarity_heat_map", "time_chart", "tsne_documents_distribution","data_map_documents_distribution",
  #             "top_5_word_scores", "top_10_word_scores","term_rank", "topic_hierarchy"]
  diagrams = ["topic_distance_map", "topic_similarity_heat_map", "time_chart", "tsne_documents_distribution","tsne_10000_documents_distribution","data_map_documents_distribution",
              "top_5_word_scores", "top_10_word_scores","term_rank", "topic_hierarchy"]
  for name in diagrams:
    if name in locals():
      diagram = locals()[name]
      if diagram is not None:
        output[name] = diagram

  return output
