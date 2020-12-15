from kaldi.util.table import SequentialMatrixReader
import os
import kaldi_io
import numpy as np

for dataset in ['train']:
    for feats in ['f0', 'en']:
        os.makedirs('dump/train/cmvn_{}'.format(feats), exist_ok=True)
        os.makedirs('dump/dev/cmvn_{}'.format(feats), exist_ok=True)
        with SequentialMatrixReader("scp:data/{}/feats.{}.0.scp".format(dataset, feats)) as feats_reader:
            kv = [(key, np.array(value)) for key, value in feats_reader]
            value = np.vstack([v[1] for v in kv])

        with open('data/{}/stats.{}'.format(dataset, feats), 'w+') as f:
            mean = np.mean(value)
            std = np.std(value)
            f.write('{} {}'.format(np.mean(value), np.std(value)**2))

        with kaldi_io.open_or_fd('ark:| copy-feats --compress=true ark:- ark,scp:dump/{}/cmvn_{}/feats.ark,data/{}/feats.{}.cmvn.scp'.format(dataset, feats, dataset, feats), 'wb') as w:
            for key, value in kv:
                kaldi_io.write_mat(w, (value-mean)/std, key=key)

        with SequentialMatrixReader("scp:data/dev/feats.{}.0.scp".format( feats)) as feats_reader:
            kv = [(key, np.array(value)) for key, value in feats_reader]
            value = np.vstack([v[1] for v in kv])
        with kaldi_io.open_or_fd('ark:| copy-feats --compress=true ark:- ark,scp:dump/dev/cmvn_{}/feats.ark,data/dev/feats.{}.cmvn.scp'.format(feats, feats), 'wb') as w:
            for key, value in kv:
                kaldi_io.write_mat(w, (value-mean)/std, key=key)



        
