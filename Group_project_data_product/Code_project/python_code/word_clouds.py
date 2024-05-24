from wordcloud import WordCloud
import pandas as pd
import os
from os.path import join as opj
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

news_df = pd.read_csv("/root/autodl-tmp/workspace/BERTopic/data/new_text_withstem.csv")
# Extract

cleaned_text = news_df['headline_cleaned_text']
original_text = news_df['headline_text']
cleaned_text = [cleaned_text[i] if isinstance(cleaned_text[i],str) else original_text[i] for i in range(len(cleaned_text))]

num_samples = 100000
data = cleaned_text[:num_samples]

def wordcloud(txt):
    #txt = " ".join(headline.lower() for headline in news_df['headline_text'])
    wordcloud = WordCloud().generate(txt)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
        
    
def get_topic_word_cloud_from_contents(topic_index,topic_npy):
  txt_list = [""] * 100000
  index = 0
  topic_array = np.load(topic_npy)
  for i in range(len(topic_array)):
    if topic_array[i] == topic_index:
      txt_list[index] = data[i]
      index += 1
  txt = " ".join(txt_list)
  return wordcloud(txt)

topic_index = 6
model_name = "google_questions_emb"
topic_npy = f"/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings/{model_name}/k_means_8_cluster__and__LSA_dr/topics.npy"
diagram = get_topic_word_cloud_from_contents(topic_index,topic_npy)

output_path = opj("/root/autodl-tmp/workspace/BERTopic/output/diagrams",f"{model_name}_topic_{topic_index}" + "word_clouds.png")
# diagram.write_image(output_path)
plt.savefig(output_path)
