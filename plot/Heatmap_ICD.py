import pandas as pd
import torch
import pickle
from plot.utils import select_top_words, select_phi_from_topics, filter_phi_from_words, plot_icd9_topics, plot_seed_topics
# import os
# import sys
# from pathlib import Path
# current_dir = Path(__file__).parent.absolute()
# root_dir = current_dir.parent
# sys.path.append(str(root_dir))

import logging
# Set the logging level to 'WARNING' to suppress debug (and info) messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# the phenotype topics we want to find, and the corresponding phecodes
topic_list = ['asthma', 'chf', 'copd', 'diabetes', 'epilepsy', 'hiv', 'hypertension', 'ihd', 'schizophrenia']
phecode_list = [495, 428, 496, 250, 345, 71, 401, 411, 295.1]

# topic_list = ['asthma', 'chf', 'copd', 'diabetes', 'epilepsy', 'hiv', 'hypertension', 'ihd']
# phecode_list = [661.0, 480.13, 281.12, 38.3, 939.0, 194.0, 395.1, 495.0]
# topic_list = ['Dementia', 'Depression', 'Dev Delay', 'Obesity', 'Schizophrenia']
# phecode_list = [290.1, 296.2, 315.0, 278.1, 295.1] # Dev delay maybe is 315 parent phecode
max_words = 3

# read data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_n = torch.load("../parameters/exp_n_icd_0.pt", map_location=device)
exp_n = exp_n.cpu().detach().numpy()
V, K = exp_n.shape
beta = 0.1
phi_r = (beta + exp_n) / (beta*V + exp_n.sum(axis=1, keepdims=1))
phi_r = phi_r / phi_r.sum(axis=0, keepdims=1) # normalization over V for each topic

# read index mapping of vocab and phecode
vocab_ids = pickle.load(open("../mapping/icd_vocab_ids.pkl", "rb")) # key is icd, value is the mapped index
inv_vocab_ids = {v: k for k, v in vocab_ids.items()}
phecode_ids = pickle.load(open("../mapping/phecode_ids.pkl", "rb")) # key is phecode, value is the mapped index
tokenized_phecode_icd = pickle.load(open("../mapping/tokenized_phecode_icd.pkl", 'rb'))  # get seed word-topic mapping, V x K matrix

# select phi from the chosen topics
select_phi_r, phecode_seed_dict = select_phi_from_topics(phi_r, V, phecode_list, phecode_ids, tokenized_phecode_icd)
# select top words from the phi
top_words = select_top_words(select_phi_r, max_words, len(topic_list))
# filter phi based on the selected top words (topic number * max_words)
filter_phi_r = filter_phi_from_words(select_phi_r, phecode_list, top_words, max_words)
# map the found icd indices to the actual icd codes
top_icds_list = [inv_vocab_ids[v] for v in top_words]


# find whether each icd code is a seed for the phecode
seed_ind_list = []
for k, seeds in phecode_seed_dict.items():
    seed_ind_list.extend([1 if top_words[k * max_words + i] in seeds else 0 for i in range(max_words)])
# map icd code to code name
icd_mapping_df = pd.read_csv('D_ICD_DIAGNOSES.csv')
# icd_name_dict = dict(zip(icd_mapping_df['ICD9_CODE'], icd_mapping_df['LONG_TITLE']))
icd_name_dict = dict(zip(icd_mapping_df['ICD9_CODE'], icd_mapping_df['SHORT_TITLE']))
top_icd_name_list = []
for i, word in enumerate(top_icds_list):
    # some icd codes (like 2765) are not in D_ICD_DIAGNOSES.csv but in the training data, we need to assign manually
    if word == '2765':
        top_icd_name_list.append(word + ' ' + 'Volume depletion')
        continue
    top_icd_name_list.append(str(word) + ' ' + icd_name_dict[word])
# plot_icd9_topics(filter_phi_r, top_icd_name_list, phecode_list, topic_list)
plot_seed_topics(filter_phi_r, top_icd_name_list, phecode_list, topic_list, seed_ind_list)
