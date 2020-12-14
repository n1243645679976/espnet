import numpy as np
for dataset in ['train', 'dev']:
    dic = {}
    dic1 = {}
    with open('data/'+dataset+'/orig_duration') as f, open('data/'+dataset+'/utt2num_frames') as f1, open('data/'+dataset+'/duration', 'w+') as w:
        for line in f.readlines():
            dic[line.split()[0]] = np.cumsum(list(map(float,line.split()[1:])))
        for line in f1.readlines():
            dic1[line.split()[0]] = float(line.split()[1])
        for key in dic.keys():
            y = dic[key]
            k = dic[key][-1]
#            print(y, k, dic1[key])
            dic[key] = list(map(lambda x: int(x/k * dic1[key]), y))
            to_write = np.hstack([dic[key][0], list(map(str, np.diff(dic[key])))])
            re_towrite = np.sum(list(map(float, to_write)))
            if re_towrite != dic1[key]:
                print('!!!!!!!!!', key, re_towrite, dic1[key], to_write)
            w.write('{} '.format(key) + ' '.join(to_write) + '\n')
        



        
