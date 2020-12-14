import kaldi_io
for dataset in ['train','dev']:
    b = {}
    c = {}
    with open('data/'+dataset+'/utt2num_frames') as f:
        for line in f.readlines():
            b[line.split()[0]] = int(line.split()[1])
    with open('data/'+dataset+'/duration') as f:
        for line in f.readlines():
            c[line.split()[0]] = list(map(int, list(line.split()[1:])))
    for key in a.keys():
        print(b[key], sum(c[key])-c[key][0], sum(c[key]) - c[key][-1])


