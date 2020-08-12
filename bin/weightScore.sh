#!/bin/bash

# ./bin/weightScore.sh  VoxCeleb1-O_eer1.7497_aamsoftmax_test.score VoxCeleb1-O_eer1.7922_amsoftmax_test.score fu.score

weight1=0.5
weight2=0.5

. bin/parse_options.sh

if [ $# != 3 ];then
echo "usage: $0 [--weight1 0.5] [--weight2 0.5] <score-file1> <score-file2> <output-score>"
exit 1
fi

label_column=1
score1=$1
score2=$2
output=$3

if [ $label_column == 3 ];then
  awk -v weight1=$weight1 -v weight2=$weight2 'NR==FNR{a[$1$2]=$3}NR>FNR{print $1,$2,$3*weight1+a[$1$2]*weight2}' $score1 $score2 > $output
elif [ $label_column == 1 ];then
  awk -v weight1=$weight1 -v weight2=$weight2 'NR==FNR{a[$2$3]=$1}NR>FNR{print $1*weight1+a[$2$3]*weight2,$2,$3}' $score1 $score2 > $output
fi

echo "Weight score done"
