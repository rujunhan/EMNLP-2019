from nltk.corpus import wordnet as wn
import numpy as np
from collections import OrderedDict
import copy
import codecs

def read_glove(input_dir):
    glove_emb = open(input_dir, 'r+', encoding="utf-8")
    emb_dict = OrderedDict([(x.strip().split(' ')[0], [float(xx) for xx in x.strip().split(' ')[1:]]) for x in glove_emb])

    return emb_dict

def create_pos_dict(nlp_ann):
    pos_dict = OrderedDict()
    for key, pos in nlp_ann.items():
        # to find the last '['                                                                                                                            
        key = str(key)
        splt = key.rfind('[')
        tok = key[:splt]
        span = key[splt:]
        ### span has to be the key because duplicate tokens can occur in a text                                                                           
        pos_dict[span] = (tok, pos.label)
    return pos_dict


def ner_features(nlp_ann, left, right):

    all_ner = {str(en.span): (str(en.text()), str(en.entity_type)) for en in nlp_ann.mentions()}
    all_ner = {range(int(k.split(':')[0][1:]), int(k.split(':')[1][:-1])):v for k,v in all_ner.items()}
    left_tlen = len(left.text.split(' '))
    right_tlen = len(right.text.split(' '))

    ner_fts = [0, 0]
    for k,v in all_ner.items():

        if left_tlen > 1 and left.span[0] in k and (left.span[1] - 1) in k and v[1] in ['DATE', 'TIME']:
            ner_fts[0] = 1
        if right_tlen > 1 and right.span[0] in k and (right.span[1] - 1) in k and v[1] in ['DATE', 'TIME']:
            ner_fts[1] = 1

    return ner_fts

def token_idx(left, right, pos_dict):

    all_keys = list(pos_dict.keys())

    ### to handle case with multiple tokens                                                                                       
    lkey_start = str(left[0])
    lkey_end = str(left[1])

    ### to handle start is not an exact match -- "tomtake", which should be "to take"                                                                                                                                                                                                                            
    lidx_start = 0
    while int(all_keys[lidx_start].split(':')[1][:-1]) <= left[0]:
        lidx_start += 1

    ### to handle case such as "ACCOUNCED--" or multiple token ends with not match                                                                                                                                                          
    lidx_end = lidx_start
    try:
        while left[1] > int(all_keys[lidx_end].split(':')[1][:-1]):
            lidx_end += 1
    except:
        lidx_end -= 1

    rkey_start = str(right[0])
    rkey_end = str(right[1])

    ridx_start = 0
    while int(all_keys[ridx_start].split(':')[1][:-1]) <= right[0]:
        ridx_start += 1

    ridx_end = ridx_start
    try:
        while right[1] > int(all_keys[ridx_end].split(':')[1][:-1]):
            ridx_end += 1
    except:
        ridx_end -= 1
    return all_keys, lidx_start, lidx_end, ridx_start, ridx_end

def compute_ngbrs(all_keys, lidx_start, lidx_end, ridx_start, ridx_end, pos_dict, pos_ngbrs, pos_fts=True):
    
    idx = int(pos_fts)
    if lidx_start < pos_ngbrs:
        lngbrs = ['<pad>' for k in range(pos_ngbrs - lidx_start)] + [pos_dict[all_keys[k]][idx] for k in list(range(lidx_start)) + list(range(lidx_end + 1, lidx_end+pos_ngbrs+1))]
    elif lidx_end > (len(all_keys) - 1 - pos_ngbrs):
        lngbrs = [pos_dict[all_keys[k]][idx] for k in list(range(lidx_start - pos_ngbrs, lidx_start)) + list(range(lidx_end + 1, len(all_keys)))] + ['<pad>' for k in range(pos_ngbrs - (len(all_keys) - 1 - lidx_end))]
    else:
        lngbrs = [pos_dict[all_keys[k]][idx] for k in list(range(lidx_start-pos_ngbrs, lidx_start)) + list(range(lidx_end + 1, lidx_end+1+pos_ngbrs))]

    assert len(lngbrs) == 2 * pos_ngbrs

    if ridx_start < pos_ngbrs:
        rngbrs = ['<pad>' for k in range(pos_ngbrs - ridx_start)] + [pos_dict[all_keys[k]][idx] for k in list(range(ridx_start)) + list(range(ridx_end + 1, ridx_end+pos_ngbrs+1))]

    elif ridx_end > len(all_keys) - pos_ngbrs - 1:
        rngbrs = [pos_dict[all_keys[k]][idx] for k in list(range(ridx_start - pos_ngbrs, ridx_start)) + list(range(ridx_end + 1, len\
(all_keys)))] + ['<pad>' for k in range(pos_ngbrs - (len(all_keys) - 1 - ridx_end))]

    else:
        rngbrs = [pos_dict[all_keys[k]][idx] for k in list(range(ridx_start-pos_ngbrs, ridx_start)) + list(range(ridx_end + 1, ridx_end+1+pos_ngbrs))]

    assert len(rngbrs) == 2 * pos_ngbrs

    return lngbrs, rngbrs


def pos_features(all_keys, lidx_start, lidx_end, ridx_start, ridx_end, pos_dict, pos_ngbrs, pos2idx):
       
    lngbrs, rngbrs = compute_ngbrs(all_keys, lidx_start, lidx_end, ridx_start, ridx_end, pos_dict, pos_ngbrs)
    return [pos2idx[x] if x in pos2idx.keys() else len(pos2idx) for x in [pos_dict[all_keys[lidx_start]][1], pos_dict[all_keys[ridx_start]][1]] + lngbrs + rngbrs]


def distance_features(lidx_start, lidx_end, ridx_start, ridx_end):

    ### if multiple tokens, take the mid-point                                                                                     
    return (float(lidx_start) + float(lidx_end)) / 2.0 -  (float(ridx_start) + float(ridx_end) ) / 2.0


def modal_features(lidx_start, lidx_end, ridx_start, ridx_end, pos_dict):

    modals = ['will', 'would', 'can', 'could', 'may', 'might']
    all_tokens = [tok.lower() for tok, span in pos_dict.values()]

    return [1 if md in all_tokens[lidx_end + 1 : ridx_start] else 0 for md in modals]

def temporal_features(lidx_start, lidx_end, ridx_start, ridx_end, pos_dict):

    temporal = ['before', 'after', 'since', 'afterwards', 'first', 'lastly', 'meanwhile', 'next', 'while', 'then']
    all_tokens = [tok.lower() for tok, span in pos_dict.values()]

    return [1 if tp in all_tokens[lidx_end + 1 : ridx_start] else 0 for tp in temporal]


def wordNet_features(lidx_start, lidx_end, ridx_start, ridx_end, pos_dict):

    all_tokens = [tok.lower() for tok, span in pos_dict.values()]

    features = []
    try:
        sims = set(wn.synsets(all_tokens[lidx_start])).intersection(wn.synsets(all_tokens[ridx_start]))
        if len(sims) > 0:
            features.append(1)
        else:
            features.append(0)
    except:
        features.append(0)

    try:
        lderiv = set(itertools.chain.from_iterable([lemma.derivationally_related_forms() for lemma in wn.lemmas(all_tokens[lidx_start])]))
        rderiv = set(itertools.chain.from_iterable([lemma.derivationally_related_forms() for lemma in wn.lemmas(all_tokens[ridx_start])]))
        if len(lderiv.intersection(rderiv))> 0:
            features.append(1)
        else:
            features.append(0)
    except:
        features.append(0)

    return features

def polarity_features(left, right):

    lp = 1.0 if left.polarity == "POS" else 0.0
    rp = 1.0 if right.polarity == "POS" else 0.0
    return [lp, rp]


def tense_features(left, right):
    
    tense_dict = {'PAST': 0, 
                  'PRESENT': 1, 
                  'INFINITIVE': 2, 
                  'FUTURE': 3, 
                  'PRESPART': 4, 
                  'PASTPART': 5}

    li = np.zeros(len(tense_dict))
    ri = np.zeros(len(tense_dict))

    if left.tense in tense_dict:
        li[tense_dict[left.tense]] = 1.0

    if right.tense in tense_dict:
        ri[tense_dict[right.tense]] = 1.0

    return list(li) + list(ri)
