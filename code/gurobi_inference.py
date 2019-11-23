from gurobipy import *
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict
from typing import Iterator, List, Mapping, Union, Optional, Set
from datetime import datetime
from utils import ClassificationReport
import numpy as np
import pickle

class Global_Inference():
    
    def __init__(self, prob_ents, prob_rels, cand_rels, label2idx, pairs, ew):
        
        self.model = Model("joint_inference")

        self.prob_ents = prob_ents
        self.prob_rels = prob_rels

        self.N, self.Nc = prob_ents.shape
        self.M, self.Mc = prob_rels.shape

        self.pred_ent_labels = list(np.argmax(prob_ents, axis=1))
        self.pred_rel_labels = list(np.argmax(prob_rels, axis=1))

        self.cand_rels = cand_rels
        self.ew = ew

        self.label2idx = label2idx
        self.idx2label = OrderedDict([(v,k) for k,v in label2idx.items()])
        
        self.pairs = pairs
        self.idx2pair = {n: self.pairs[n] for n in range(len(pairs))}
        self.pair2idx = {v:k for k,v in self.idx2pair.items()}

    def define_vars(self):
        var_table_e, var_table_r = [], []

        # entity variables
        for n in range(self.N):
            sample = []
            for p in range(self.Nc):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="e_%s_%s"%(n,p)))
            var_table_e.append(sample)

        # relation variables
        for m in range(self.M):
            sample = []
            for p in range(self.Mc):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="r_%s_%s"%(m,p)))
            var_table_r.append(sample)

        return var_table_e, var_table_r
        
    def objective(self, samples_e, samples_r, p_table_e, p_table_r):
    
        obj = 0.0

        assert len(samples_e) == self.N 
        assert len(samples_r) == self.M
        assert len(samples_e[0]) == self.Nc
        assert len(samples_r[0]) == self.Mc
        
        # entity
        for n in range(self.N):
            for p in range(self.Nc):
                obj += self.ew * samples_e[n][p] * p_table_e[n][p]

        # relation
        for m in range(self.M):
            for p in range(self.Mc):
                obj += samples_r[m][p] * p_table_r[m][p]

        return obj
    
    def single_label(self, sample):
        return sum(sample) == 1

    def rel_ent_sum(self, samples_e, samples_r, e, r, c):
        # negative rel constraint
        return samples_e[e[0]][0] + samples_e[e[1]][0] - samples_r[r][c]

    def rel_left_ent(self, samples_e, samples_r, e, r, c):
        # positive rel left constraint
        return samples_e[e[0]][1] - samples_r[r][c]
        
    def rel_right_ent(self, samples_e, samples_r, e, r, c):
        # positive rel right constraint
        return samples_e[e[1]][1] - samples_r[r][c]
        
    def transitivity_list(self):

        transitivity_samples = []
        pair2idx = self.pair2idx

        for k, (e1, e2) in self.idx2pair.items():
            for (re1, re2), i in pair2idx.items():
                if e2 == re1 and (e1, re2) in pair2idx.keys():
                    transitivity_samples.append((pair2idx[(e1, e2)], pair2idx[(re1, re2)], pair2idx[(e1, re2)]))
        return transitivity_samples

    def transitivity_criteria(self, samples, triplet):
        # r1  r2  Trans(r1, r2)                                                                                    
        # _____________________                                                                                    
        # r   r   r                                                                                                
        # r   s   r                                                                                                
        # b   v   b, v                                                                                             
        # a   v   a, v                                                                                             
        # v   b   b, v                                                                                             
        # v   a   a, v                                                                                             
        r1, r2, r3 = triplet
        label_dict = self.label2idx
        
        return [
            samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']],
            samples[r1][label_dict['AFTER']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']],
            samples[r1][label_dict['SIMULTANEOUS']] + samples[r2][label_dict['SIMULTANEOUS']] - samples[r3][label_dict['SIMULTANEOUS']],
            #samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['VAGUE']],
            #samples[r1][label_dict['NONE']] + samples[r2][label_dict['NONE']] - samples[r3][label_dict['NONE']],
            samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']], 
            samples[r1][label_dict['AFTER']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']],
            samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']],
            samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']]
        ]

        ''' TBD
        return [
                samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']],
                samples[r1][label_dict['AFTER']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']],
                samples[r1][label_dict['SIMULTANEOUS']] + samples[r2][label_dict['SIMULTANEOUS']] - samples[r3][label_dict['SIMULTANEOUS']],
                samples[r1][label_dict['INCLUDES']] + samples[r2][label_dict['INCLUDES']] - samples[r3][label_dict['INCLUDES']],
                samples[r1][label_dict['IS_INCLUDED']] + samples[r2][label_dict['IS_INCLUDED']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['VAGUE']],        
                samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['INCLUDES']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['INCLUDES']],
                samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['IS_INCLUDED']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['AFTER']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['AFTER']] + samples[r2][label_dict['INCLUDES']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['INCLUDES']],
                samples[r1][label_dict['AFTER']] + samples[r2][label_dict['IS_INCLUDED']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['INCLUDES']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']],
                samples[r1][label_dict['INCLUDES']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']],
                samples[r1][label_dict['INCLUDES']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['AFTER']],
                samples[r1][label_dict['IS_INCLUDED']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['IS_INCLUDED']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['AFTER']],
                samples[r1][label_dict['IS_INCLUDED']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['IS_INCLUDED']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['IS_INCLUDED']],
                samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['INCLUDES']] - samples[r3][label_dict['INCLUDES']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['AFTER']],
                samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['IS_INCLUDED']] - samples[r3][label_dict['IS_INCLUDED']] - samples[r3][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['AFTER']]
               ]
        '''
    def define_constraints(self, var_table_e, var_table_r):
        # Constraint 1: single label assignment
        for n in range(self.N):
            self.model.addConstr(self.single_label(var_table_e[n]), "c1_%s" % n)
        for m in range(self.M):
            self.model.addConstr(self.single_label(var_table_r[m]), "c1_%s" % (self.N + m))

        # Constraint 2: Positive relation requires positive event arguments
        for r, cr in enumerate(self.cand_rels):
            for c in range(self.Mc-1):
                self.model.addConstr(self.rel_left_ent(var_table_e, var_table_r, cr, r, c) >= 0, "c2_%s_%s" % (r, c))
                self.model.addConstr(self.rel_right_ent(var_table_e, var_table_r, cr, r, c) >= 0, "c3_%s_%s" % (r, c))
            if c == self.Mc-1:
                self.model.addConstr(self.rel_ent_sum(var_table_e, var_table_r, cr, r, c) >= 0, "c4_%s_%s" % (r, c))
        
        
        # Constraint 3: transitivity                                                                                   
        trans_triples = self.transitivity_list()
        t = 0
        for triple in trans_triples:
            for ci in self.transitivity_criteria(var_table_r, triple):
                self.model.addConstr(ci <= 1, "c5_%s" % t)
                t += 1
        return 
    
    def run(self):
        try:
            # Define variables
            var_table_e, var_table_r = self.define_vars()

            # Set objective 
            self.model.setObjective(self.objective(var_table_e, var_table_r, self.prob_ents, 
                                                   self.prob_rels), GRB.MAXIMIZE)
            
            # Define constrains
            self.define_constraints(var_table_e, var_table_r)

            # run model
            self.model.setParam('OutputFlag', False)
            self.model.optimize()
            
        except GurobiError:
            print('Error reported')


    def predict(self):
        ent_count, rel_count = 0, 0

        for i, v in enumerate(self.model.getVars()):
            
            # rel_ent indicator
            is_ent = True if v.varName.split('_')[0] == 'e' else False
            # sample idx
            s_idx = int(v.varName.split('_')[1])
            # sample class index
            c_idx = int(v.varName.split('_')[2])

            if is_ent:
                if v.x == 1.0 and self.pred_ent_labels[s_idx] != c_idx:
                    #print(v.varName, self.pred_ent_labels[s_idx])
                    self.pred_ent_labels[s_idx] = c_idx
                    ent_count += 1
            else:
                if v.x == 1.0 and self.pred_rel_labels[s_idx] != c_idx:
                    #print(v.varName, self.pred_rel_labels[s_idx])
                    self.pred_rel_labels[s_idx] = c_idx
                    rel_count += 1

        print('# of global entity correction: %s' % ent_count)
        print('# of global relation correction: %s' % rel_count)
        print('Objective Function Value:', self.model.objVal)
        
        return 
