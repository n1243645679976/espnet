from espnet2.layers.global_mvn import GlobalMVN
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy

from scipy.io import wavfile

import kaldi_io
import threading
import os
import torch

fs = 24000
n_fft = 1024
hop_length = 128
nj = 1

def read_scp_return_dic(scp):
    dic = {}
    with open(scp) as f:
        for line in f.readlines():
            key = line.split()[0]
            value = ' '.join(line.split()[1:])
            dic[key] = value
    return dic

dio = Dio(fs=fs, n_fft=n_fft, hop_length=hop_length)
ener = Energy(fs=fs, n_fft=n_fft, hop_length=hop_length)
os.makedirs('f0', exist_ok=True)
os.makedirs('en', exist_ok=True)
for dataset in [ 'train']:
    os.makedirs('f0/'+dataset, exist_ok=True)
    os.makedirs('en/'+dataset, exist_ok=True)

    num_frames = read_scp_return_dic('data/'+dataset+'/utt2num_frames') 
    wavscp = read_scp_return_dic('data/'+dataset+'/wav.scp')
    duration = read_scp_return_dic('data/'+dataset+'/duration')
    
    def extract(keys, num):
        with kaldi_io.open_or_fd('ark:| copy-feats --compress=true ark:- ark,scp:f0/'+dataset+'/feats.f0.{}.ark,data/'.format(num)+dataset+'/feats.f0.{}.scp'.format(num), 'wb') as w, kaldi_io.open_or_fd('ark:| copy-feats --compress=true ark:- ark,scp:en/'+dataset+'/feats.en.{}.ark,data/'.format(num)+dataset+'/feats.en.{}.scp'.format(num), 'wb') as w2, open('data/'+dataset+'/pitch_len', 'w+') as w1, open('data/'+dataset+'/energy_len', 'w+') as w3:
          f0mvn = []
          enmvn = []
          for key in keys:
            sr, wav = wavfile.read(wavscp[key])
            wav = torch.from_numpy(wav).float().reshape(1, -1)/32768
            lll = torch.tensor([list(map(int, duration[key].split()))])
            x, il = dio(wav, [int(wav.shape[1])], [lll.sum()], lll, [len(duration[key].split())])
            f0mvn.append(torch.mean(x))
            x = x.reshape(-1, 1).numpy()
            kaldi_io.write_mat(w, x, key=key)
            w1.write('{} {}\n'.format(key, il))


            llll = torch.tensor([list(map(int, duration[key].split()))])
            x, il = ener(wav, torch.tensor([int(wav.shape[1])]), [llll.sum()], llll, [len(duration[key].split())])
            enmvn.append(torch.mean(x))
            x =x.reshape(-1, 1).numpy()
            kaldi_io.write_mat(w2, x, key=key)
            w3.write('{} {}\n'.format(key, il))
    extract(list(wavscp.keys()), 0)
        
#    threads = []
#    for i in range(nj):
#        threads.append(threading.Thread(target=extract, args=(list(wavscp.keys())[i::nj], i)))
#        threads[i].start()
#    for i in range(nj):
#        threads[i].join()




        

            


