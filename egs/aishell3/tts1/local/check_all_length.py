from kaldi.util.table import SequentialMatrixReader
for dataset in ['train', 'dev']:
    dudic = {}
    with open('data/'+dataset+'/duration') as f:
        for line in f.readlines():
            key = line.split()[0]
            value = len(line.split()[1:])
            dudic[key] = value
    with SequentialMatrixReader("scp:data/"+dataset+'/feats.f0.0.scp') as feats_reader:
        f0dic = {key:value.shape for key, value in feats_reader}
    with SequentialMatrixReader("scp:data/"+dataset+'/feats.en.0.scp') as feats_reader:
        endic = {key:value.shape for key, value in feats_reader}
    for key in f0dic.keys():
        if f0dic[key][0] != endic[key][0]:
            print('f0 != enc on key', key)
        if f0dic[key][0] != dudic[key]:
            print('f0 != du on key', key)




