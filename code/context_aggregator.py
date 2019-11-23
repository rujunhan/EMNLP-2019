import pickle
import argparse
import os
from collections import OrderedDict
def main(args):

    for split in ["train", "dev", "test"]:

        with open('%s/all_joint/%s.pickle' % (args.data_dir, split), 'rb') as handle:
            data = pickle.load(handle)
        handle.close()

        context_map = OrderedDict([])
        count = 0
        for ex in data:
            
            start = ex[4][1][1][0]
            end = ex[4][1][-2][0]
            
            # use doc_id + start token and end token spans as unique context id
            context_id = (ex[0], start, end)
            
            # sample id, (left id, right id), label_idx, distance, reverse_ind,
            # (left_start, left_end, right_start, right_end), pred_ind
            rel = (ex[1], ex[2], ex[3], ex[4][3], ex[4][4], ex[4][5:9], ex[4][9])
            if context_id in context_map:
                context_map[context_id]['rels'].append(rel)
            else:
                context_map[context_id] = {'context_id': count,
                                           'doc_id': ex[0],
                                           'context': ex[4][0:3], #bert, event_label, pos
                                           'rels': [rel]}
                count += 1

        save_dir = args.data_dir + '/all_context/'
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open('%s/%s.pickle' % (save_dir, split), 'wb') as handle:
            pickle.dump(context_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    return



if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', type=str)
    p.add_argument('-data_type', type=str, default="matres")

    args = p.parse_args()
    
    args.data_dir += args.data_type
    main(args)
