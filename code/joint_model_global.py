from pathlib import Path
import pickle
import sys
import argparse
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import logging as log
import abc
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
import math
import time
import copy
import json
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from featureFuncs import *
from functools import partial
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from joint_model import pad_collate, EventDataset, BertClassifier
from gurobi_inference_rel import Global_Inference

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(123)

@dataclass()
class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()

    def predict(self, model, data, args, test=False, gold=False, model_r=None):
        model.eval()
        
        criterion = nn.CrossEntropyLoss()                                                                           
        count = 1
        labels_r, probs_r, losses, losses_e = [], [], [], []
        pred_inds = []

        # stoare non-predicted rels in list
        nopred_rels = []

        ent_pred_map, ent_label_map, ent_idx_map = {}, {}, {}
        all_ent_key, all_ent_pos, all_label_e, all_prob_e = [], [], [], []

        all_pairs, all_label_r, all_prob_r = [], [], []
        all_lidx_start, all_ridx_start = [], []
        l_idx, r_idx, context_start = 0, 0, 0

        for doc_id, context_id, sents, ent_keys, ents, poss, rels, lengths in data:

            if args.cuda:
                sents = sents.cuda()
                ents = ents.cuda()

            ## predict entity first                                                                     
            out_e, prob_e = model(sents, lengths, task='entity')
            ## construct candidate relations                                                                                             
            rel_label, fts, rel_idxs, doc_id, pairs, lidx_start, lidx_end, ridx_start, ridx_end, none_rel  \
                = self.construct_relations(prob_e, lengths, rels, list(doc_id), poss, gold=False, 
                                           ent_thresh=args.ent_thresh, rel_thresh=args.rel_thresh, ent_keys=ent_keys, test=test)

            nopred_rels.extend(none_rel)

            if args.cuda:
                rel_label = rel_label.cuda()
                fts = fts.cuda()

            all_label_r.append(rel_label)
            all_pairs.extend([(doc + "_" + x, doc + "_" + y) for doc, (x, y) in zip(doc_id, pairs)])

            ## predict relations                                                                                                  
            out_r, prob_r = model(sents, lengths, fts=fts, rel_idxs=rel_idxs, lidx_start=lidx_start,
                                  lidx_end=lidx_end, ridx_start=ridx_start, ridx_end=ridx_end)
            #docs.extend(doc)
            #pairs.extend(pair)

            loss_e = []
            ## global inference for each unique context                                                                                  
            b_str = 0
            pred_ent_labels, pred_rel_labels = [], []
            for b, l in enumerate(lengths):
                all_lidx_start.extend([x + context_start for x in lidx_start[b]])
                all_ridx_start.extend([x + context_start for x in ridx_start[b]])
                context_start += l

                all_prob_e.append(prob_e[b, :l, :])
                all_label_e.append(ents[b, :l])

                # flatten entity key - a list of original (extend)                                                        
                assert len(ent_keys[b]) == l
                all_ent_key.extend([p for p in ent_keys[b]])
                # flatten pos tags                                                                                        
                all_ent_pos.extend([p for p in poss[b]])
            all_prob_r.append(prob_r)

        all_prob_e = torch.cat(all_prob_e)
        all_label_e = torch.cat(all_label_e)
        assert all_prob_e.size(0) == all_label_e.size(0)
        assert all_label_e.size(0) == len(all_ent_key)

        all_label_r = torch.cat(all_label_r)
        all_prob_r = torch.cat(all_prob_r)
        assert len(all_pairs) == all_prob_r.size(0)
        assert len(all_pairs) == all_label_r.size(0)
        assert len(all_pairs) == len(all_lidx_start)
        assert len(all_pairs) == len(all_ridx_start)

        # global inference for relation with transitivity                                                     
        best_pred_idx_e, best_pred_idx_r, pred_ent_labels, pred_rel_labels \
            = self.global_prediction(all_prob_e, all_prob_r, all_lidx_start, 
                                     all_ridx_start, all_pairs, args.entity_weight, evaluate=True)
        # Compute Loss
        loss_r = self.loss_func_rel(best_pred_idx_r, all_label_r, all_prob_r, args.margin)
        loss_e = self.loss_func_ent(best_pred_idx_e, all_label_e, all_prob_e, args.margin)
        loss = args.relation_weight * loss_r + args.entity_weight * loss_e

        all_label_e = all_label_e.tolist()                                                                                                                                                                               
        for i, v in enumerate(all_ent_key):                                                                                                           
            label_e = all_label_e[i]                                                                                                                
                                                                                                                                                      
            # exclude sent_start and sent_sep                                                                                                     
            if v in ["[SEP]", "[CLS]"]:                                                                                                           
                assert all_ent_pos[i] in ["[SEP]", "[CLS]"]                                                                                           
                                                                                                                                                      
            if v not in ent_pred_map:                                                                                                             
                # store global assignments                                                                                                        
                ent_pred_map[v] = [pred_ent_labels[i]]                                                                                            
                ent_label_map[v] = (label_e, all_ent_pos[i])                                                                                          
            else:                                                                                                                                 
                # if key stored already, append another prediction                                                                                
                ent_pred_map[v].append(pred_ent_labels[i])                                                                                        
                # and ensure label is the same                                                                                                    
                assert ent_label_map[v][0] == label_e                                                                                             
                assert ent_label_map[v][1] == all_ent_pos[i]

        assert all_label_r.size(0) == len(pred_rel_labels)

        # calculate entity F1 score here
        # update ent_pred_map with [mean > 0.5 --> 1]

        ent_pred_map_agg = {k:1 if np.mean(v) >= 0.5 else 0 for k,v in ent_pred_map.items()}
        #ent_pred_map_agg = {k:max(v) for k,v in ent_pred_map.items()}
        n_correct = 0
        n_pred = 0
            
        pos_keys = OrderedDict([(k, v) for k, v in ent_label_map.items() if v[0] == 1])
        n_true = len(pos_keys)

        for k,v in ent_label_map.items():
            if ent_pred_map_agg[k] == 1:
                n_pred += 1
            if ent_pred_map_agg[k] == 1 and ent_label_map[k][0] == 1:
                n_correct += 1

        print(n_pred, n_true, n_correct)

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else float(numr) / float(denr)

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)
        
        print("Evaluation temporal relation loss: %.4f" % loss_r.data)
        
        print("Evaluation temporal entity loss: %.4f; F1: %.4f" % (loss_e.data, f1_score))

        if test:
            return pred_rel_labels, all_label_r, f1_score, nopred_rels
        else:
            return pred_rel_labels, all_label_r, n_pred, n_true, n_correct, nopred_rels

    def construct_relations(self, prob_e, lengths, rels, doc, poss, gold=True, ent_thresh = 0.1, rel_thresh = 0.5, ent_keys=[], test=False):
        # many relation properties such rev and pred_ind are not used for now

        ## Case 1: only use gold relation
        if gold:
            pred_rels = rels

        ## Case 2: use candidate relation predicted by entity model
        else:
            def _is_gold(pred_span, gold_rel_span):
                return ((gold_rel_span[0] <= pred_span <= gold_rel_span[1]))

            # gold labels not predicted, should be none, sanity check
            nopred_rels = []
            
            batch_size = len(lengths)
            
            # eliminate ent_pred > context length
            # add filter for events with certain pos tags (based on train set)
            #include_pos = [6, 11, 12, 26, 27, 28, 29, 30, 31] # tbd
            include_pos = [26, 27, 28, 29, 30, 31] # matres
            #include_pos = [36, 5, 6, 4, 11, 12, 19, 26, 27, 28, 29, 30, 31]
            ent_locs = [[x for x in range(l) if poss[b][x] in include_pos and prob_e[b, x, 1] > ent_thresh] 
                        for b,l in enumerate(lengths)]

            # all possible relation candiate based on pred_ent
            rel_locs = [list(combinations(el, 2)) for el in ent_locs]

            pred_rels = []
            totl = 0
            # use the smallest postive sample id as start of neg id
            # this may not be perfect, but we really don't care about neg id
            neg_counter = min([int(x[0][1:]) for rel in rels for x in rel])
            
            for i, rl in enumerate(rel_locs):
                temp_rels, temp_ids = [], []
                for r in rl:
                    # filtered with both events has local prob < 0.5
                    if prob_e[i, r[0], 1] < rel_thresh and prob_e[i, r[1], 1] < rel_thresh:
                        #print(i, r)
                        continue
                    sent_segs = len([x for x in poss[i] if x == '[SEP]'])
                    in_seg = [x for x in poss[i][r[0] : r[1]] if x == '[SEP]']
                    ### exclude rel that are in the same sentence, but two segments exist. i.e. unique input context
                    if (sent_segs > 1) and (len(in_seg) == 0):
                        continue
                    else:
                        if test:
                            self.final_output['e1_prob'].append(prob_e[i][r[0]].tolist()[1])
                            self.final_output['e2_prob'].append(prob_e[i][r[1]].tolist()[1])
                            self.final_output['e1_span'].append(ent_keys[i][r[0]][1])
                            self.final_output['e2_span'].append(ent_keys[i][r[1]][1])
                            self.final_output['doc_id'].append(doc[i])
                        
                        totl += 1
                        gold_match = [x for x in rels[i] if _is_gold(r[0], x[5][:2]) and _is_gold(r[1], x[5][2:])]
                        # multiple tokens could indicate the same events. 
                        # simple pick the one occurs first
                        if len(gold_match) > 0 and gold_match[0][0] not in temp_ids:
                            temp_rels.append(gold_match[0])
                            temp_ids.append(gold_match[0][0])
                            if test:
                                self.final_output['e1_id'].append(gold_match[0][1][0])
                                self.final_output['e2_id'].append(gold_match[0][1][1])
                                self.final_output['gold_rel'].append(gold_match[0][2])
                        else:
                            ## construct a negative relation pair -- 'NONE'
                            neg_id = 'N%s' % neg_counter
                            left_match = [x for x in rels[i] if _is_gold(r[0], x[5][:2])]
                            right_match = [x for x in rels[i] if _is_gold(r[1], x[5][2:])]
                            # provide a random but unique id for event predicted if not matched in gold
                            left_id = left_match[0][1][0] if len(left_match) > 0 else ('n%s' % (neg_counter + 10000))
                            right_id = right_match[0][1][1] if len(right_match) > 0 else ('n%s' % (neg_counter + 20000))
                            a_rel = (neg_id, (left_id, right_id), self._label_to_id['NONE'],
                                     [float(r[1] - r[0])], False, (r[0], r[0], r[1], r[1]), True)
                            temp_rels.append(a_rel)
                            neg_counter += 1
                            if test:
                                self.final_output['e1_id'].append(left_id)
                                self.final_output['e2_id'].append(right_id)
                                self.final_output['gold_rel'].append(self._label_to_id['NONE'])
                            
                nopred_rels.extend([x[2] for x in rels[i] if x[0] not in [tr[0] for tr in temp_rels]])

                if test:
                    for x in rels[i]:
                        if x[0] not in [tr[0] for tr in temp_rels]:
                            self.final_output_np['gold_rel'].append(x[2])
                            self.final_output_np['pred_rel'].append(self._label_to_id['NONE'])
                            self.final_output_np['e1_id'].append(x[1][0])
                            self.final_output_np['e2_id'].append(x[1][1])
                            self.final_output_np['e1_prob'].append(prob_e[i][x[5][0]].tolist()[1])
                            self.final_output_np['e2_prob'].append(prob_e[i][x[5][3]].tolist()[1])
                            self.final_output_np['e1_span'].append(ent_keys[i][x[5][0]][1])
                            self.final_output_np['e2_span'].append(ent_keys[i][x[5][3]][1])
                            self.final_output_np['doc_id'].append(doc[i])
                #assert len(nopred_rels) == 0
                # B * N_b
                pred_rels.append(temp_rels)

        # relations are (flatten) lists of features
        # rel_idxs indicates (batch_id, rel_in_batch_id)
        docs, pairs = [], []
        rel_idxs, lidx_start, lidx_end, ridx_start, ridx_end = [],[],[],[],[]
        for i, rel in enumerate(pred_rels):
            rel_idxs.extend([(i, ii) for ii, _ in enumerate(rel)])
            lidx_start.append([x[5][0] for x in rel])
            lidx_end.append([x[5][1] for x in rel])
            ridx_start.append([x[5][2] for x in rel])
            ridx_end.append([x[5][3] for x in rel])
            pairs.extend([x[1] for x in rel])
            docs.extend([doc[i] for _ in rel])
        assert len(docs) == len(pairs)
            
        rels = [x for rel in pred_rels for x in rel]
        if rels == []:
            labels = torch.FloatTensor([])
            fts = torch.FloatTensor([])
        else:
            labels = torch.LongTensor([x[2] for x in rels])
            fts = torch.cat([torch.FloatTensor(x[3]) for x in rels]).unsqueeze(1)
        
        return labels, fts, rel_idxs, docs, pairs, lidx_start, lidx_end, ridx_start, ridx_end, nopred_rels


    def global_prediction(self, prob_table_ents, prob_table_rels, lidx, ridx, pairs, ew, evaluate=False, true_labels=[]):
        # input (for each context):                                                                    
        # 1. prob_table_ents: (context) local event predictions: N * 2, N: number of entities
        # 2. prob_table_rels: (context) local rel candidate predictions: M (# of can rels) * R (# of rel class)
        # 3. left idx: a vector of length M - left entity index
        # 4. right idx: a vector of length M - right entity index
        # 5. pairs: relation pairs for transivity rule
        # output:                                                                             
        # 1. global_ent_idx: best global assignment of entity in matrix form                          
        # 2. global_rel_idx: best global assignment of relation in matrix form

        # initialize entity table 
        N, Nc = prob_table_ents.shape
        global_ent_idx = np.zeros((N, Nc), dtype=int)

        # initialize relation table
        M, Mc = prob_table_rels.shape
        global_rel_idx = np.zeros((M, Mc), dtype=int)

        cand_rels = list(zip(lidx, ridx))
        assert M == len(cand_rels)
        
        global_model = Global_Inference(prob_table_ents.detach().numpy(), 
                                        prob_table_rels.detach().numpy(), 
                                        cand_rels, self._label_to_id, pairs, ew)
        global_model.run()
        global_model.predict()

        # entity global assignment 
        for n in range(N):
            global_ent_idx[n, global_model.pred_ent_labels[n]] = 1
        
        # relation global assignment
        for m in range(M):
            global_rel_idx[m, global_model.pred_rel_labels[m]] = 1

        if evaluate:
            #assert len(true_labels) == N + M
            #global_model.evaluate(true_labels)
            return global_ent_idx, global_rel_idx, global_model.pred_ent_labels, global_model.pred_rel_labels
        else:
            return global_ent_idx, global_rel_idx

    def _train(self, train_data, eval_data, pos_emb, args):

        model = BertClassifier(args)

        if args.cuda:
            print("using cuda device: %s" % torch.cuda.current_device())
            assert torch.cuda.is_available()
            model.cuda()

        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr = args.lr, momentum=args.momentum, weight_decay=args.decay)
        #criterion = nn.CrossEntropyLoss()                                                                                                                                                
        losses = [] 

        sents, poss, ftss, labels = [], [], [], []                                                                            
        if args.load_model == True:
            checkpoint = torch.load(args.ilp_dir + args.load_model_file, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint['epoch']
            best_eval_f1 = checkpoint['f1']
            print("Local best eval f1 is: %s" % best_eval_f1)
                                                                     
        best_eval_f1 = 0.0 
        best_epoch = 0

        for epoch in range(args.epochs):
            print("Training Epoch #%s..." % epoch)
            model.train()
            count = 1

            loss_hist_t, loss_hist_e = [], []

            start_time = time.time()

            gold = False #if epoch > args.pipe_epoch else True
            #event_pos_counter = Counter()

            all_pairs, all_label_r, all_prob_r = [], [], []
            all_label_e, all_prob_e = [], []

            # record unique context start idx for both rel and ent
            #context_ent, context_rel = [], []
            all_lidx_start, all_ridx_start = [], []
            l_idx, r_idx, context_start = 0, 0, 0
            for doc_id, context_id, sents, keys, ents, poss, rels, lengths in train_data:

                if args.cuda:
                    sents = sents.cuda()
                    ents = ents.cuda()

                model.zero_grad() 
                     
                ## entity detection
                out_e, prob_e = model(sents, lengths, task='entity')

                ## construct candidate relations
                rel_label, fts, rel_idxs, doc_id, pairs, lidx_start, lidx_end, ridx_start, ridx_end, non_preds \
                    = self.construct_relations(prob_e, lengths, rels, list(doc_id), poss, gold=False)

                if args.cuda:
                    rel_label = rel_label.cuda()
                    fts = fts.cuda()
                
                all_label_r.append(rel_label)
                all_pairs.extend([(doc + "_" + x, doc + "_" + y) for doc, (x, y) in zip(doc_id, pairs)])
                
                ## predict relations
                out_r, prob_r = model(sents, lengths, fts=fts, rel_idxs=rel_idxs, lidx_start=lidx_start, 
                                      lidx_end=lidx_end, ridx_start=ridx_start, ridx_end=ridx_end)                    

                ## collect all unique contexts for joint inference                 
                for b, l in enumerate(lengths):
                    all_lidx_start.extend([x + context_start for x in lidx_start[b]])
                    all_ridx_start.extend([x + context_start for x in ridx_start[b]])
                    context_start += l

                    all_prob_e.append(prob_e[b, :l, :])
                    all_label_e.append(ents[b, :l])
                
                all_prob_r.append(prob_r)
                count += 1

            all_prob_e = torch.cat(all_prob_e)
            all_label_e = torch.cat(all_label_e)
            assert all_prob_e.size(0) == all_label_e.size(0)
            
            all_label_r = torch.cat(all_label_r)
            all_prob_r = torch.cat(all_prob_r)
            assert len(all_pairs) == all_prob_r.size(0)
            assert len(all_pairs) == all_label_r.size(0)
            assert len(all_pairs) == len(all_lidx_start)
            assert len(all_pairs) == len(all_ridx_start)

            # global inference for relation with transitivity
            best_pred_idx_e, best_pred_idx_r = self.global_prediction(all_prob_e, all_prob_r, all_lidx_start, 
                                                                      all_ridx_start, all_pairs, args.entity_weight)
            loss_r = self.loss_func_rel(best_pred_idx_r, all_label_r, all_prob_r, args.margin)
            loss_e = self.loss_func_ent(best_pred_idx_e, all_label_e, all_prob_e, args.margin)
            
            # combine
            #loss = args.relation_weight * loss_r + args.entity_weight * loss_e
            loss = loss_r + loss_e
            loss.backward()
                
            optimizer.step() 
                
            if args.cuda:
                if loss_r != 0:
                    loss_hist_t.append(loss_r.data.cpu().numpy())
                    loss_hist_e.append(loss_e.data.cpu().numpy())
            else:
                if loss_r != 0:
                    loss_hist_t.append(loss_r.data.numpy())                                        
                    loss_hist_e.append(loss_e.data.numpy())
            print("Temporal loss is %.4f" % np.mean(loss_hist_t))
            print("Entity loss is %.4f" % np.mean(loss_hist_e))
            print("%.4f seconds elapsed" % (time.time() - start_time))                                                              
            # Evaluate at the end of each epoch                                                                              
            print("*"*50)

            if len(eval_data) > 0:

                # need to have a warm-start otherwise there could be no event_pred
                # may need to manually pick poch < #, but 0 generally works when ew is large
                #eval_gold = True if epoch == 0 else args.eval_gold
                eval_gold = gold
                pred_labels, eval_labels, ent_pred, ent_true, ent_corr, nopred_rels = self.predict(model, eval_data, args, gold=eval_gold)

                eval_labels = list(eval_labels.numpy())
                assert len(eval_labels) == len(pred_labels)

                pred_labels.extend([self._label_to_id['NONE'] for _ in nopred_rels])
                eval_labels.extend(nopred_rels)
                # select model only based on entity + relation F1 score 
                eval_f1 = self.weighted_f1(pred_labels, eval_labels, ent_corr, ent_pred, ent_true, 
                                           args.relation_weight, args.entity_weight)

                # args.pipe_epoch <= args.epochs if pipeline (joint) training is used 
                if eval_f1 > best_eval_f1 and (epoch > args.pipe_epoch or args.pipe_epoch >= 1000):
                    best_eval_f1 = eval_f1
                    self.model = copy.deepcopy(model)
                    best_epoch = epoch

                print("Evaluation F1: %.4f" % (eval_f1))
                print("*"*50)
                
        print("Final Evaluation F1: %.4f at Epoch %s" % (best_eval_f1, best_epoch))
        print("*"*50)

        if args.epochs == 0:
            pred_labels, eval_labels, ent_pred, ent_true, ent_corr, nopred_rels = self.predict(model, eval_data, args, gold=False)

            eval_labels = list(eval_labels.numpy())
            assert len(eval_labels) == len(pred_labels)

            pred_labels.extend([self._label_to_id['NONE'] for _ in nopred_rels])
            eval_labels.extend(nopred_rels)
            # select model only based on entity + relation F1 score                                                                    
            eval_f1 = self.weighted_f1(pred_labels, eval_labels, ent_corr, ent_pred, ent_true,
                                       args.relation_weight, args.entity_weight)

            # args.pipe_epoch <= args.epochs if pipeline (joint) training is used                                                      
            if eval_f1 > best_eval_f1 and (epoch > args.pipe_epoch or args.pipe_epoch >= 1000):
                best_eval_f1 = eval_f1
                self.model = copy.deepcopy(model)
                best_epoch = epoch

        if args.save_model == True:
            torch.save({'epoch': epoch,
                        'args': args,
                        'state_dict': self.model.state_dict(),
                        'f1': best_eval_f1,
                        'optimizer' : optimizer.state_dict()
                    }, "%s%s.pth.tar" % (args.ilp_dir, args.save_stamp))
        
        return best_eval_f1, best_epoch
                          
    def loss_func_ent(self, best_pred_idx_e, all_label_e, prob_e, margin):

        ## max prediction scores                                                          
        mask_e = torch.ByteTensor(best_pred_idx_e)

        assert mask_e.size() == prob_e.size()

        max_score_e = torch.masked_select(prob_e, mask_e)

        #globalNlocal = (probs.data.max(1)[0].view(-1) != max_scores.data.view(-1)).numpy()

        ## Entity true label scores                                                                     
        N, Nc = prob_e.size()
        idx_mat_e = np.zeros((N, Nc), dtype=int)

        for n in range(N):
            idx_mat_e[n][all_label_e[n]] = 1
        mask_e = torch.ByteTensor(idx_mat_e)
        assert mask_e.size() == prob_e.size()
        label_score_e = torch.masked_select(prob_e, mask_e)

        ## Entity SSVM loss
        # distance measure: try Hamming Distance later
        #delta = torch.FloatTensor([margin for _ in range(N)])
        delta = Variable(torch.FloatTensor([0.00000001 if label_score_e[n].data == max_score_e[n].data else margin for n in range(N)]), requires_grad=True)
        diff = delta + max_score_e - label_score_e

        # loss should be non-negative                                                                                               
        losses_e = []
        for n in range(N):
            if diff[n].data.numpy() <= 0.0:
                losses_e.append(Variable(torch.FloatTensor([0.0])))
            else:
                losses_e.append(diff[n].reshape(1,))

        return torch.mean(torch.cat(losses_e))

    
    def loss_func_rel(self, best_pred_idx_r, all_label_r, prob_r, margin):
        
        mask_r = torch.ByteTensor(best_pred_idx_r)
        assert mask_r.size() == prob_r.size()
        max_score_r = torch.masked_select(prob_r, mask_r)

        ## Relation true label scores
        M, Mc = prob_r.size()

        idx_mat_r = np.zeros((M, Mc), dtype=int)

        for m in range(M):
            idx_mat_r[m][all_label_r[m]] = 1

        mask_r = torch.ByteTensor(idx_mat_r)
        assert mask_r.size() == prob_r.size()
        label_score_r = torch.masked_select(prob_r, mask_r)

        ## Relation loss                                                                                                        
        #delta = torch.FloatTensor([margin for _ in range(M)])
        delta = Variable(torch.FloatTensor([0.00000001 if label_score_r[m].data == max_score_r[m].data else margin for m in range(M)]), requires_grad=True)
        diff = delta + max_score_r - label_score_r

        count = 0
        losses_r = []
        for m in range(M):
            if diff[m].data.numpy() <= 0.0:
                losses_r.append(Variable(torch.FloatTensor([0.0])))
            else:
                count += 1
                losses_r.append(diff[m].reshape(1,))

        return torch.mean(torch.cat(losses_r))


    def train_epoch(self, train_data, dev_data, args, test_data = None):

        if args.data_type == "matres":
            label_map = matres_label_map
        if args.data_type == "tbd":
            label_map = tbd_label_map
        
        assert len(label_map) > 0

        all_labels = list(OrderedDict.fromkeys(label_map.values()))
        ## append negative pair label
        all_labels.append('NONE')

        if args.joint:
            label_map_c = causal_label_map
            # in order to perserve order of unique keys
            all_labels_c =  list(OrderedDict.fromkeys(label_map_c.values()))
            self._label_to_id_c = OrderedDict([(all_labels_c[l],l) for l in range(len(all_labels_c))])
            self._id_to_label_c = OrderedDict([(l,all_labels_c[l]) for l in range(len(all_labels_c))])
            print(self._label_to_id_c)
            print(self._label_to_id_c)

        self._label_to_id = OrderedDict([(all_labels[l],l) for l in range(len(all_labels))])
        self._id_to_label = OrderedDict([(l,all_labels[l]) for l in range(len(all_labels))])

        print(self._label_to_id)
        print(self._id_to_label)

        args.label_to_id = self._label_to_id

        ### pos embdding is not used for now, but can be added later
        pos_emb= np.zeros((len(args.pos2idx) + 1, len(args.pos2idx) + 1))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0

        best_f1, best_epoch = self._train(train_data, dev_data, pos_emb, args)
        print("Final Dev F1: %.4f" % best_f1)
        return best_f1, best_epoch

    def weighted_f1(self, pred_labels, true_labels, ent_corr, ent_pred, ent_true, rw=0.0, ew=0.0):
        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        assert len(pred_labels) == len(true_labels)

        weighted_f1_scores = {}
        if 'NONE' in self._label_to_id.keys():
            num_tests = len([x for x in true_labels if x != self._label_to_id['NONE']])
        else:
            num_tests = len([x for x in true_labels])

        print("Total relation to evaluate: %s" % len(true_labels))
        print("Total positive relation samples to eval: %s" % num_tests)
        total_true = Counter(true_labels)
        total_pred = Counter(pred_labels)

        labels = list(self._id_to_label.keys())

        n_correct = 0
        n_true = 0
        n_pred = 0

        if rw > 0:
            # f1 score is used for tcr and matres and hence exclude vague                            
            exclude_labels = ['NONE', 'VAGUE'] if len(self._label_to_id) == 5 else ['NONE']

            for label in labels:
                if self._id_to_label[label] not in exclude_labels:

                    true_count = total_true.get(label, 0)
                    pred_count = total_pred.get(label, 0)

                    n_true += true_count
                    n_pred += pred_count

                    correct_count = len([l for l in range(len(pred_labels))
                                         if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
                    n_correct += correct_count
        if ew > 0:
            # add entity prediction results before calculating precision, recall and f1
            n_correct += ent_corr
            n_pred += ent_pred
            n_true += ent_true

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)
        print("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))

        return(f1_score)

@dataclass
class EventEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, test_data: Iterator[FlatRelation], args):
        # load test data first since it needs to be executed twice in this function                                                
        print("start testing...")

        pred_labels, true_labels, ent_f1, nopred_rels \
            = self.model.predict(self.model.model, test_data, args, test = True, gold = False)

        
        pred_labels.extend([self.model._label_to_id['NONE'] for _ in nopred_rels])
        true_labels = true_labels.tolist()
        true_labels.extend(nopred_rels)
        
        rel_f1 = self.model.weighted_f1(pred_labels, true_labels, 0, 0, 0, rw=1.0)

        print("Gold pairs labled as None: %s" % len(nopred_rels))

        pred_labels = [self.model._id_to_label[x] for x in pred_labels]
        true_labels = [self.model._id_to_label[x] for x in true_labels]
        
        print(len(pred_labels), len(true_labels))

        out = ClassificationReport(args.model, true_labels, pred_labels)
        print(out)
        print("F1 Excluding Vague: %.4f" % rel_f1)
        return rel_f1, ent_f1

def main(args):

    data_dir = args.data_dir
    opt_args = {}

    params = {'batch_size': args.batch,
              'shuffle': False,
              'collate_fn': pad_collate}

    type_dir = "/all_context/"
    test_data = EventDataset(args.data_dir + type_dir, "test")
    test_generator = data.DataLoader(test_data, **params)

    train_data = EventDataset(args.data_dir + type_dir, "train")
    train_generator = data.DataLoader(train_data, **params)

    dev_data = EventDataset(args.data_dir + type_dir, "dev")
    dev_generator = data.DataLoader(dev_data, **params)
    
    model = NNClassifier()
    print(f"======={args.model}=====\n")
    best_f1, best_epoch = model.train_epoch(train_generator, dev_generator, args)
    evaluator = EventEvaluator(model)
    rel_f1, ent_f1 = evaluator.evaluate(test_generator, args)
    
    print(rel_f1, ent_f1)

    return 
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # arguments for data processing
    p.add_argument('-data_dir', type=str, default = '../data')
    p.add_argument('-other_dir', type=str, default = '../other')
    # select model
    p.add_argument('-model', type=str, default='joint/global') #'multitask/gold', 'multitask/pipeline'
    # arguments for RNN model
    p.add_argument('-emb', type=int, default=100)
    p.add_argument('-hid', type=int, default=90)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-batch', type=int, default=1)
    p.add_argument('-data_type', type=str, default="red")
    p.add_argument('-epochs', type=int, default=3)
    p.add_argument('-pipe_epoch', type=int, default=1000) # 1000: no pipeline training; otherwise <= epochs 
    p.add_argument('-seed', type=int, default=123)
    p.add_argument('-lr', type=float, default=0.1) # 0.0005, 0.001
    p.add_argument('-num_classes', type=int, default=2) # get updated in main()
    p.add_argument('-dropout', type=float, default=0.6)
    p.add_argument('-pos2idx', type=dict, default = {})
    p.add_argument('-w2i', type=OrderedDict)
    p.add_argument('-glove', type=OrderedDict)
    p.add_argument('-cuda', action='store_true')
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-n_fts', type=int, default=1)
    p.add_argument('-relation_weight', type=float, default=1.0)
    p.add_argument('-entity_weight', type=float, default=1.0)
    p.add_argument('-save_model', type=bool, default=False)
    p.add_argument('-save_stamp', type=str, default="relation_best")
    p.add_argument('-load_model_file', type=str, default="matres_pipeline_best.pt")
    p.add_argument('-joint', type=bool, default=False) # Note: this is for tcr causal pairs 
    p.add_argument('-load_model', type=bool, default=True)
    p.add_argument('-num_causal', type=int, default=2)
    p.add_argument('-bert_config', type=dict, default={})
    p.add_argument('-loss_u', type=str, default="")
    p.add_argument('-fine_tune', type=bool, default=False)
    p.add_argument('-eval_with_timex', type=str, default=False)
    p.add_argument('-eval_gold',type=bool, default=False)
    p.add_argument('-margin', type=float, default=0.3)
    p.add_argument('-momentum', type=float, default=0.9)
    p.add_argument('-decay', type=float, default=0.9)
    p.add_argument('-ent_thresh',type=float, default=0.2)
    p.add_argument('-rel_thresh',type=float, default=0.5)
    args = p.parse_args()

    #args.eval_gold = True if args.pipe_epoch >= 1000 else False

    # if training with pipeline, ensure train / eval pipe epoch are the same
    #if args.pipe_epoch < 1000:
    #    assert args.pipe_epoch == args.eval_pipe_epoch

    args.data_dir += args.data_type
    # create pos_tag and vocabulary dictionaries
    # make sure raw data files are stored in the same directory as train/dev/test data
    tags = open("/nas/home/rujunhan/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx
    
    args.idx2pos = {v+1:k for k,v in pos2idx.items()}

    args.bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size_or_config_json_file": 30522
    }
    print(args.momentum, args.decay)
    main(args)

    
