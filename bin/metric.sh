#!/bin/bash

#  ./bin/metric.sh  veri_test.txt fu.score

trials=$1
calib_file=$2

python tools/VoxSRC2020/compute_min_dcf.py  --p-target 0.05 --c-miss 1 --c-fa 1 $calib_file  $trials
python tools/VoxSRC2020/compute_EER.py  --ground_truth $trials --prediction  $calib_file
python tools/voices_scorer/score_voices --label-column 1 --p-target 0.05 $calib_file $trials
