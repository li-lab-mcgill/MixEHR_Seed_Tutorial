# For each document, compute the count of ICD code under each PheCodes
import torch
import pickle
import os
import sys
from pathlib import Path
current_dir = Path(__file__).parent.absolute()
root_dir = current_dir.parent
sys.path.append(str(root_dir))
from corpus import Corpus

vocab_ids = pickle.load(open("../mapping/icd_vocab_ids.pkl", "rb"))
inv_vocab_ids = {v: k for k, v in vocab_ids.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds_topic_matrix = torch.load("../phecode_mapping/seed_topic_matrix.pt", map_location=device)  # get seed word-topic mapping, V x K matrix
V, K = seeds_topic_matrix.shape
c = Corpus.read_corpus_from_directory('../store/', 'corpus.pkl') # read corpus file
print('obtain D x K document-PheCode count matrix')
document_phecode_matrix = torch.zeros((c.D, K), device=device) # document-PheCode counts, D x K matrix
pat_d = []
for d_i, doc in enumerate(c.dataset):
    pat_d.append(doc.doc_id)
    for v, freq in doc.words_dict[0].items():
        document_phecode_matrix[d_i] += seeds_topic_matrix[v] * freq
    print(doc, torch.sum(document_phecode_matrix[d_i]))
torch.save(document_phecode_matrix, "../guide_prior/document_phecode_matrix.pt")
