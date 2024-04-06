import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=1)


def select_top_words(phi, max_words, K):
    top_words = []
    for k in range(K):
        phi_k = phi[:, k]
        sorted_word = np.argsort(-phi_k, axis=0)
        top_words.extend(list(sorted_word[:max_words]))
    return top_words


def select_phi_from_topics(phi_r, V, phecode_list, phecode_ids, tokenized_phecode_icd):
    select_phi_r = np.zeros((V, len(phecode_list)))  # V X K(=11)
    phecode_seed_dict = {}
    for k, phecode in enumerate(phecode_list):
        phecode_seed_dict[k] = []
        if int(phecode) == phecode:
            icd_sum = 0
            child_phecode_dict = {}
            for key, value in phecode_ids.items():
                if value == 0:
                    continue
                if int(key) == int(phecode):
                    child_phecode_dict[value] = tokenized_phecode_icd[value]
                    icd_sum += len(tokenized_phecode_icd[value])
            for key, value in child_phecode_dict.items():
                phecode_seed_dict[k].extend(value)
                select_phi_r[:, k] += len(value) / icd_sum * phi_r[:, key]
        else:
            phecode_seed_dict[k].extend(tokenized_phecode_icd[phecode_ids[phecode]])
            select_phi_r[:, k] = phi_r[:, phecode_ids[phecode]]
    return select_phi_r, phecode_seed_dict


def select_phi_from_multiple_topics(phi_r, V, phecode_list, phecode_ids, tokenized_phecode_icd):
    select_phi_r = np.zeros((V, len(phecode_list)))  # V X K
    phecode_seed_dict = {}
    for k, phecodes in enumerate(phecode_list):
        child_phecode_dict = {}
        phecode_seed_dict[k] = []
        for phecode in phecodes:
            if int(phecode) == phecode:
                icd_sum = 0
                for key, value in phecode_ids.items(): # iterate over all phecodes to find child phecodes
                    if value == 0:
                        continue
                    if int(key) == int(phecode):
                        child_phecode_dict[value] = tokenized_phecode_icd[value]
                        icd_sum += len(tokenized_phecode_icd[value])
                for key, value in child_phecode_dict.items():
                    phecode_seed_dict[k].extend(value)
                    if icd_sum <2:
                        select_phi_r[:, k] += (len(value) / (icd_sum + 5)) * phi_r[:, key]
                    else:
                        select_phi_r[:, k] += (len(value) / (icd_sum)) * phi_r[:, key]
                    select_phi_r[:, k] += len(value) / icd_sum * phi_r[:, key]
            else:
                phecode_seed_dict[k].extend(tokenized_phecode_icd[phecode_ids[phecode]])
                select_phi_r[:, k] = phi_r[:, phecode_ids[phecode]]
    return select_phi_r, phecode_seed_dict

def select_phi_from_multiple_subtopics(phi_r, V, phecode_list, phecode_ids, tokenized_phecode_icd):
    select_phi_r = np.zeros((V, len(phecode_list) * len(phecode_list[0])))  # V X (K*phecode_per_label)
    phecode_seed_dict = {}
    for k, phecodes in enumerate(phecode_list):
        phecode_seed_dict[k] = []
        for i, phecode in enumerate(phecodes):
            phecode_seed_dict[k].append(tokenized_phecode_icd[phecode_ids[phecode]])
            select_phi_r[:, k * len(phecode_list[0]) + i] = phi_r[:, phecode_ids[phecode]]
    return select_phi_r, phecode_seed_dict

def select_phi_from_topics(phi_r, V, phecode_list, phecode_ids, tokenized_phecode_icd):
    select_phi_r = np.zeros((V, len(phecode_list)))  # V X K(=11)
    phecode_seed_dict = {}
    for k, phecode in enumerate(phecode_list):
        phecode_seed_dict[k] = []
        if int(phecode) == phecode:
            icd_sum = 0
            child_phecode_dict = {}
            for key, value in phecode_ids.items():
                if value == 0:
                    continue
                if int(key) == int(phecode):
                    child_phecode_dict[value] = tokenized_phecode_icd[value]
                    icd_sum += len(tokenized_phecode_icd[value])
            for key, value in child_phecode_dict.items():
                phecode_seed_dict[k].extend(value)
                select_phi_r[:, k] += len(value) / icd_sum * phi_r[:, key]
        else:
            phecode_seed_dict[k].extend(tokenized_phecode_icd[phecode_ids[phecode]])
            select_phi_r[:, k] = phi_r[:, phecode_ids[phecode]]
    return select_phi_r, phecode_seed_dict

def filter_phi_from_words(select_phi_r, phecode_list, top_words, max_words):
    filter_phi_r = select_phi_r[top_words, :]
    return filter_phi_r

def plot_icd9_topics(filter_phi, word_list, phecode_list, topic_list):
    font_color = '#525252'
    hfont = {'fontname': 'Calibri'}
    facecolor = '#eaeaf2'
    class_name = ['Infectious and parasitic diseases', 'Neoplasms',
                  'Endocrine, nutritional and metabolic diseases, and immunity disorders',
                  'Diseases of the blood and blood-forming organs', 'Mental disorders',
                  'Diseases of the nervous system', 'Diseases of the sense organs',
                  'Diseases of the circulatory system', 'Diseases of the respiratory system',
                  'Diseases of the digestive system', 'Diseases of the genitourinary system',
                  'Complications of pregnancy, childbirth, and the puerperium',
                  'Diseases of the skin and subcutaneous tissue',
                  'Diseases of the musculoskeletal system and connective tissue', 'Congenital anomalies',
                  'Certain conditions originating in the perinatal period',
                  'Symptoms, signs, and ill-defined conditions', 'Injury and poisoning',
                  'External causes of injury',
                  'Supplementary classification of factors influencing health status and contact with health services', ]
    ICD_partition = ['0', '140', '240', '280', '290', '320', '360', '390', '460', '520', '580', '630', '680',
                          '710', '740', '760', '780', '800', 'E', 'V']  # 20 classes
    labels = [np.searchsorted(ICD_partition, w, side='right')-1 for w in word_list]
    class_label = np.arange(len(class_name))
    network_pal = sns.cubehelix_palette(len(class_label), light=0.95, dark=0.05, start=1, rot=-2)
    class_lut = dict(zip(class_label, network_pal))
    class_colors = list(map(class_lut.get, labels))
    # x_label_list = [str(phecode) + ' (' + pheno + ')' for phecode, pheno in zip(phecode_list, topic_list)]
    x_label_list = topic_list
    # x_label_list = phecode_list
    phi_data = pd.DataFrame(data=filter_phi, columns=x_label_list, index=word_list)
    g = sns.clustermap(phi_data, row_cluster=False, col_cluster=False, row_colors=class_colors, vmax=0.25, vmin=0,
                       linewidth=2.5, cmap="BuPu", yticklabels=True, cbar_pos=(0, .2, .03, .4))
    for label in class_label:
        g.ax_col_dendrogram.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    # g.ax_col_dendrogram.legend(loc='lower right', ncol=1)
    g.cax.set_visible(False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=13, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=11.5)
    fig, ax = plt.subplots()
    plt.scatter(x=[0], y=[0])
    for label in np.unique(labels):
        ax.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    ax.legend(loc='best', facecolor='white', framealpha=1)
    # plt.show()
    plt.savefig('ICD.png')

def plot_icd10_topics(filter_phi, word_list, phecode_list, topic_list):
    '''
    plot heatmap of topics in ICD-10 modality
    '''
    font_color = '#525252'
    hfont = {'fontname': 'Calibri'}
    facecolor = '#eaeaf2'
    class_name = ['Certain infectious and parasitic diseases',
                  'Neoplasms',
                  'Diseases blood involving the immune mechanism',
                  'Endocrine, nutritional and metabolic diseases',
                  'Mental and behavioural disorders',
                  'Diseases nervous system',
                  'Diseases eye and adnexa',
                  'Diseases ear and mastoid process',
                  'Diseases circulatory system',
                  'Diseases respiratory system',
                  'Diseases digestive system',
                  'Diseases skin and subcutaneous tissue',
                  'Diseases musculoskeletal system and connective tissue',
                  'Diseases genitourinary system',
                  'Pregnancy, childbirth and the puerperium',
                  'Certain conditions originating in the perinatal period',
                  'Congenital malformations and chromosomal abnormalities',
                  'Symptoms, signs and abnormal clinical and lab.findings',
                  'Injury, poisoning and consequences of external causes',
                  'External causes of morbidity and mortality',
                  'Factors influencing health status'] # 21 classes
    labels = []
    index_labels = []
    for w in word_list:
        if w[0] == 'A' or w[0] == 'B':
            labels.append(class_name[0])
            index_labels.append(0)
        elif w[0] == 'C' or (w[0] == 'D' and int(w[1:3]) <= 48):
            labels.append(class_name[1])
            index_labels.append(1)
        elif (w[0] == 'D' and int(w[1:3]) >= 50):
            labels.append(class_name[2])
            index_labels.append(2)
        elif w[0] == 'E':
            labels.append(class_name[3])
            index_labels.append(3)
        elif w[0] == 'F':
            labels.append(class_name[4])
            index_labels.append(4)
        elif w[0] == 'G':
            labels.append(class_name[5])
            index_labels.append(5)
        elif (w[0] == 'H' and int(w[1:3]) <= 59):
            labels.append(class_name[6])
            index_labels.append(6)
        elif (w[0] == 'H' and int(w[1:3]) >= 60):
            labels.append(class_name[7])
            index_labels.append(7)
        elif w[0] == 'I':
            labels.append(class_name[8])
            index_labels.append(8)
        elif w[0] == 'J':
            labels.append(class_name[9])
            index_labels.append(9)
        elif w[0] == 'K':
            labels.append(class_name[10])
            index_labels.append(10)
        elif w[0] == 'L':
            labels.append(class_name[11])
            index_labels.append(11)
        elif w[0] == 'M':
            labels.append(class_name[12])
            index_labels.append(12)
        elif w[0] == 'N':
            labels.append(class_name[13])
            index_labels.append(13)
        elif w[0] == '0':
            labels.append(class_name[14])
            index_labels.append(14)
        elif w[0] == 'P':
            labels.append(class_name[15])
            index_labels.append(15)
        elif w[0] == 'Q':
            labels.append(class_name[16])
            index_labels.append(16)
        elif w[0] == 'R':
            labels.append(class_name[17])
            index_labels.append(17)
        if w[0] == 'S' or w[0] == 'T':
            labels.append(class_name[18])
            index_labels.append(18)
        if w[0] == 'V' or w[0] == 'W' or w[0] == 'X' or w[0] == 'Y':
            labels.append(class_name[19])
            index_labels.append(19)
        elif w[0] == 'Z':
            labels.append(class_name[20])
            index_labels.append(20)

    class_label = np.arange(len(class_name))
    network_pal = sns.cubehelix_palette(len(class_label), light=0.95, dark=0.05, start=1, rot=-2)
    class_lut = dict(zip(class_label, network_pal))
    class_colors = list(map(class_lut.get, index_labels))
    x_label_list = [str(phecode) + ' (' + pheno + ')' for phecode, pheno in zip(phecode_list, topic_list)]
    phi_data = pd.DataFrame(data=filter_phi, columns=x_label_list, index=word_list)
    g = sns.clustermap(phi_data, row_cluster=False, col_cluster=False, row_colors=class_colors, vmax=0.5, vmin=0,
                       linewidth=2.5, cmap="BuPu", yticklabels=True, cbar_pos=(0, .2, .03, .4))
    for label in class_label:
        g.ax_col_dendrogram.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    # g.ax_col_dendrogram.legend(loc='lower right', ncol=1)
    g.cax.set_visible(False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=13)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=11.5)
    fig, ax = plt.subplots()
    plt.scatter(x=[0], y=[0])
    for label in np.unique(index_labels):
        ax.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    ax.legend(loc='best', facecolor='white', framealpha=1)
    # plt.show()
    plt.savefig('ICD.png')


def plot_seed_topics(filter_phi, word_list, phecode_list, topic_list, seed_list):
    '''
    plot heatmap of topics with respect to the seed words
    '''
    font_color = '#525252'
    hfont = {'fontname': 'Calibri'}
    facecolor = '#eaeaf2'
    class_name = ['Seed codes', 'Regular codes', ] # 2 classes
    labels = []
    index_labels = []
    for s in seed_list:
        if s == 1:
            labels.append(class_name[0])
            index_labels.append(0)
        elif s == 0:
            labels.append(class_name[1])
            index_labels.append(1)
    class_label = np.arange(len(class_name))
    network_pal = sns.cubehelix_palette(len(class_label), light=0.95, dark=0.05, start=1, rot=-2)
    class_lut = dict(zip(class_label, network_pal))
    class_colors = list(map(class_lut.get, index_labels))
    x_label_list = [str(phecode) + ' (' + pheno + ')' for phecode, pheno in zip(phecode_list, topic_list)]
    # x_label_list = topic_list
    phi_data = pd.DataFrame(data=filter_phi, columns=x_label_list, index=word_list)
    # Plot the clustermap
    g = sns.clustermap(phi_data, row_cluster=False, col_cluster=False, row_colors=class_colors, vmax=0.2, vmin=0,
                       linewidth=2.5, cmap="BuPu", yticklabels=True, figsize=(8, 8), cbar_pos=(0, .2, .03, .4))
    for label in class_label:
        g.ax_col_dendrogram.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    # g.ax_col_dendrogram.legend(loc='lower right', ncol=1)
    g.cax.set_visible(False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=13, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=11.5)
    # fig, ax = plt.subplots()
    # plt.scatter(x=[0], y=[0])
    # for label in np.unique(index_labels):
    #     ax.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    # x_label_list = [str(phecode) + ' (' + pheno + ')' for phecode, pheno in zip(phecode_list, topic_list)]
    x_label_list = topic_list
    xticks_labels = x_label_list
    plt.xticks(np.arange(len(topic_list)) + .5, labels=xticks_labels, rotation=90)
    yticks_labels = word_list
    plt.yticks(np.arange(len(word_list)) + .5, labels=yticks_labels, rotation=0)
    # ax.legend(loc='best', facecolor='white', framealpha=1)
    # plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=1.15)
    plt.savefig('ICD.png')


def plot_unguide_topics(filter_phi, word_list, phecode_list, topic_list):
    font_color = '#525252'
    hfont = {'fontname': 'Calibri'}
    facecolor = '#eaeaf2'
    fig, ax = plt.subplots(figsize=(6, 11), facecolor=facecolor)
    ax = sns.heatmap(filter_phi,
                     cmap='BuPu',
                     vmin=0,
                     vmax=0.03,
                     linewidth=0.2,
                     cbar_kws={'shrink': .72},
                     cbar=False)
    x_label_list = [str(phecode) + ' (' + pheno + ')' for phecode, pheno in zip(phecode_list, topic_list)]
    # xticks_labels = x_label_list
    ax.set_xticklabels(x_label_list, rotation=90)
    ax.set_yticklabels(word_list, fontsize=13)
    ax.tick_params(left=False, right=True, top=False, labelleft=False, labelright=True, labeltop=False,
                   labelrotation=0)
    ax.set_xticklabels(x_label_list, rotation=90, fontsize=11.5)
    # plt.xticks(np.arange(len(topic_list)) + .5, labels=xticks_labels, rotation=90)
    # yticks_labels = word_list
    # plt.yticks(np.arange(len(word_list)) + .5, labels=yticks_labels, rotation=0)
    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set(fontsize=15, color=font_color, **hfont)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=12, labelcolor=font_color)
    # plt.show()
    plt.tight_layout()
    # plt.subplots_adjust(top=1.25)
    plt.savefig('medication.png')


def plot_disease_drug_topics(filter_phi, word_list, phecode_list, topic_list, seed_list):
    '''
    plot heatmap of topics in medication modalitiy, and the side bar indicates whether it is linked drug for the phenotype
    '''
    font_color = '#525252'
    hfont = {'fontname': 'Calibri'}
    facecolor = '#eaeaf2'
    class_name = ['Linked drugs', 'Unlinked drugs', ] # 2 classes
    labels = []
    index_labels = []
    for s in seed_list:
        if s == 1:
            labels.append(class_name[0])
            index_labels.append(0)
        elif s == 0:
            labels.append(class_name[1])
            index_labels.append(1)
    class_label = np.arange(len(class_name))
    network_pal = sns.cubehelix_palette(len(class_label), light=0.95, dark=0.05, start=1, rot=-2)
    class_lut = dict(zip(class_label, network_pal))
    class_colors = list(map(class_lut.get, index_labels))
    x_label_list = [str(phecode) + ' (' + pheno + ')' for phecode, pheno in zip(phecode_list, topic_list)]
    phi_data = pd.DataFrame(data=filter_phi, columns=x_label_list, index=word_list)
    g = sns.clustermap(phi_data, row_cluster=False, col_cluster=False, row_colors=class_colors, vmax=0.15, vmin=0,
                       linewidth=2.5, cmap="BuPu", yticklabels=True, cbar_pos=(0, .2, .03, .4))
    for label in class_label:
        g.ax_col_dendrogram.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    # g.ax_col_dendrogram.legend(loc='lower right', ncol=1)
    g.cax.set_visible(False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=13)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=11.5)
    fig, ax = plt.subplots()
    plt.scatter(x=[0], y=[0])
    for label in np.unique(index_labels):
        ax.bar(0, 0, color=class_lut[label], label=class_name[label], linewidth=0)
    ax.legend(loc='best', facecolor='white', framealpha=1)
    # plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=1.15)
    plt.savefig('drug.png')
