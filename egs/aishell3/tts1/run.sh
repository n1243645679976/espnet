#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4
stop_stage=4
ngpu=1       # number of gpu in training
nj=64        # number of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000      # sampling frequency
fmax=7600       # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=128   # number of shift points
win_length="" # window length

# config file
train_config=conf/tuning/train_fastspeech2.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/60

# exp tag
tag="" # tag for managing experiments.
gpu=""

. utils/parse_options.sh || exit 1;
export CUDA_VISIBLE_DEVICES=${gpu}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
eval_set=test_clean

#if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
# Get TextGrid
    
#fi

#if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
# To phoneme
# phone
#fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    fbankdir=fbank
    for x in train dev; do
        ls data/*/* | while read line; do cat ${line} | sort > ${line}1; mv ${line}1 ${line}; done
        utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${x} \
            exp/make_fbank/${x} \
            ${fbankdir}
    done

#    utils/combine_data.sh data/${train_set}_org data/train_clean_100 data/train_clean_360
#    utils/combine_data.sh data/${dev_set}_org data/dev_clean

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
#    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set} data/${train_set}
#    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${dev_set} data/${dev_set}

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
#    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
#        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

#dict=data/lang_1char/${train_set}_units.txt
#echo "dictionary: ${dict}"
#if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
#    echo "stage 2: Dictionary and Json Data Preparation"
#    mkdir -p data/lang_1char/
#    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
#    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
#    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
#    wc -l ${dict}

    # make json labels
#    data2json.sh --feat ${feat_tr_dir}/feats.scp \
#         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
#    data2json.sh --feat ${feat_dt_dir}/feats.scp \
#         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
#    data2json.sh --feat ${feat_ev_dir}/feats.scp \
#         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
#fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${dev_set} ; do
        utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
        utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
    done

    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    for name in ${train_set} ${dev_set} ; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    # Update json
#    for name in ${train_set} ${dev_set} ; do
#        local/update_json.sh ${dumpdir}/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
#    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=data/${train_set}/data.json
    dt_json=data/${dev_set}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${dev_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp data/${name}/sub_dev_data.json ${outdir}/${name}/sub_dev_data.json
        splitjson.py --parts ${nj} ${outdir}/${name}/sub_dev_data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis"
    pids=() # initialize pids
    for name in ${dev_set} ; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${name} \
            ${outdir}_denorm/${name}/log \
            ${outdir}_denorm/${name}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
