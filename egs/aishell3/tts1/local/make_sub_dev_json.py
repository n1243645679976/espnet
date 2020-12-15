## input:
##   1. fbank
##   2. x-vector
##   3. duration
## output:
##   1. text
##   2. token
##   3. spell
##   4. tokenid
##  
##

import kaldi_io
import json
from kaldi.util.table import SequentialMatrixReader

def read_scp_return_dic(scp):
    dic = {}
    shape = {}
    with open(scp) as f:
        if scp.split('.')[-1] == 'scp':
            if 'xvector' in scp:
                shape = {key:value.shape for key, value in kaldi_io.read_vec_flt_scp(scp)}
            if 'f0' in scp:
                with SequentialMatrixReader("scp:"+scp) as feats_reader:
                    dic = {key:value for key, value in feats_reader}
                shape = {key:value.shape for key, value in dic.items()}
            elif 'en' in scp:
                with SequentialMatrixReader("scp:"+scp) as feats_reader:
                    dic = {key:value for key, value in feats_reader}
                shape = {key:value.shape for key, value in dic.items()}
            elif 'feats' in scp:
                shape = {key:value.shape for key, value in kaldi_io.read_mat_scp(scp)}
            for line in f.readlines():
                dic[line.split()[0]] = ' '.join(line.split()[1:])
        else:
            for line in f.readlines():
                dic[line.split()[0]] = ' '.join(line.split()[1:])
                shape[line.split()[0]] = len(line.split())-1
    return dic, shape

r = 0
for dataset in ['dev']:
    # input
    f0, f0_shape = read_scp_return_dic('data/'+dataset+'/feats.f0.0.scp')
    en, en_shape = read_scp_return_dic('data/'+dataset+'/feats.en.0.scp')
    fbank_dic, fbank_shape_dic = read_scp_return_dic('data/'+dataset+'/feats.scp')
    xvector_dic, xvector_shape_dic = read_scp_return_dic('exp/xvector_nnet_1a/xvectors_'+dataset+'/xvector.scp')
    duration, duration_shape = read_scp_return_dic('data/'+dataset+'/duration')

    # output
    text, text_shape = read_scp_return_dic('data/'+dataset+'/text')
    token, token_shape = read_scp_return_dic('data/'+dataset+'/token')
    spell, spell_shape = read_scp_return_dic('data/'+dataset+'/spell')
    tokenid, tokenid_shape = read_scp_return_dic('data/'+dataset+'/tokenid')

    # write info into json
    json_dic = {'utts':{}}
    for key in fbank_dic.keys():
        r+= 1
        if r>=100:
            break
        input1_dic = {"feat": fbank_dic[key], "name": "input1", "shape": [int(fbank_shape_dic[key][0]), int(fbank_shape_dic[key][1])]}
        input2_dic = {'feat': xvector_dic[key], "name": "input2", "shape": [int(xvector_shape_dic[key][0])]}
        input3_dic = {'duration': duration[key], 'name': 'duration'}
        input4_dic = {'name': 'f0', 'feat': f0[key]}
        input5_dic = {'name': 'en', 'feat': en[key]}
        input_list = [input1_dic, input2_dic, input3_dic, input4_dic, input5_dic]

        output1_dic = {'name': 'output1', 'shape': [int(tokenid_shape[key]), 150], 'text': text[key], 'token': token[key], 'spell': spell[key], 'tokenid':tokenid[key]}
        output_list = [output1_dic]
        json_dic['utts'][key] = {'input': input_list, 'output': output_list}

    with open('data/'+dataset+'/sub_dev_data.json', 'w+') as w:
        json.dump(json_dic, w, indent=4, sort_keys=True, ensure_ascii=False)

