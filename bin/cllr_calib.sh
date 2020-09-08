#!/bin/bash


# ./bin/cllr_calib.sh --trails veri_test.txt --score_files "VoxCeleb1-O_eer1.7497_aamsoftmax_test.score VoxCeleb1-O_eer1.7922_amsoftmax_test.score"

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

root=cllr_calib
tag="defualt"
model=model.pt # model ckpt
trails=trails  # SV trails
score_files=  #list of str or str
calib_file=score_${tag}.calib # calibration scores

log "$0 $*"
. ./bin/parse_options.sh

python $root/calibrate_scores.py \
  --label-column 1 \
  --log-llr \
  --save-model $model \
  $trails \
  $score_files

python $root/apply_calibration.py  \
  --label-column 1 \
  --log-llr  \
  $model  \
  $score_files \
  $calib_file

./bin/metric.sh $trails $calib_file
