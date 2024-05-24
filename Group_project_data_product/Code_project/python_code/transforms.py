import os
from os.path import join as opj
import shutil
import numpy as np

def get_csvs(input_root,subdirname, output_root):
    os.makedirs(output_root, exist_ok = True)
    dirs = os.listdir(input_root)
    for directory in dirs:
        if directory.startswith("."):
            continue
        
        csv_path = opj(opj(input_root,directory,subdirname,"topic_representations.csv"))
        output_path = opj(output_root,f"{directory[:-4]}.csv")
        shutil.copyfile(csv_path, output_path)

# input_root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_embeddings"
# csv_output_root = "/root/autodl-tmp/workspace/BERTopic/output/topics_csv/"
# subdirname = "HDBSCAN_cluster__and__umap_dr"
# output_root = opj(csv_output_root,"embeddings", subdirname)

# get_csvs(input_root,subdirname, output_root)

# from model_pools import umap_dr, LSA_dr

def get_reduced_embeddings(input_root, dr_model):
    embeddings = np.load(opj(input_root, "embeddings.npy"))
    reduced_embeddings = dr_model.fit_transform(embeddings)
    print(reduced_embeddings.shape)
    output_path = opj(input_root, "reduced_embeddings.npy")
    np.save(output_path, reduced_embeddings)

# input_root1 = "/root/autodl-tmp/workspace/BERTopic/input/umap"
# dr_model1 = umap_dr
# input_root2 = "/root/autodl-tmp/workspace/BERTopic/input/LSA"
# dr_model2 = LSA_dr
# # get_reduced_embeddings(input_root1, dr_model1)
# get_reduced_embeddings(input_root2, dr_model2)

def get_diagrams_out(input_root, output_root):
    os.makedirs(output_root, exist_ok = True)
    input_root_length = len(input_root)
    for root, directories,files in os.walk(input_root):
        for f in files:
            file_path = opj(root, f)
            relative_path = file_path[input_root_length:]
            is_diagram = f.endswith(".png")
            # for keyword in ["diagram", "graph", "maps"]:
            #     if keyword in root:
            #         is_diagram = True
            #         break
            # print(f, is_diagram)
            if is_diagram:
                output_path = opj(output_root, relative_path)
                print(output_path)

                os.makedirs(os.path.dirname(output_path), exist_ok = True)
                shutil.copyfile(file_path, output_path)

input_root = "/root/autodl-tmp/workspace/BERTopic/output/100000_test_clustering/"
output_root = "/root/autodl-tmp/workspace/BERTopic/output/clustering_diagrams"
get_diagrams_out(input_root, output_root)


