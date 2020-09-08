#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
from calibrate_scores import LinearModel


def sigmoid(x):
    return torch.sigmoid(torch.tensor(x, dtype=torch.float32)).numpy()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Apply calibration model to LLR scores")
  parser.add_argument('--label-column', metavar='label_column', type=int, default=3, choices=[1,3], help="label column")
  parser.add_argument('--log-llr', action='store_true', default=False, help="convert score to [0, 1] and log")
  parser.add_argument('model')
  parser.add_argument('input_score_file', nargs='+', help="One or more input score files. Each line is a triple <enrolled_speaker> <test_speaker> <LLR_score>") 
  parser.add_argument('output_score_file')

  args = parser.parse_args()
  
  model = LinearModel(len(args.input_score_file))
  model.load_state_dict(torch.load(args.model))
  model.eval()
  model.double()
  
  input_tensor = None
  for input_score_file in args.input_score_file:
    input_keys_and_scores = []
    for l in open(input_score_file):
      ss = l.split()
      if args.label_column == 3:
        key = ss[0] + " " + ss[1]
        score = float(ss[2])
      elif args.label_column == 1:
        key = ss[1] + " " + ss[2]
        score = float(ss[0])
      else:
        raise Exception("Illegal label column: %d" % args.label_column)
      if args.log_llr:
        #score = (score + 1) / 2
        score = sigmoid(score)
        score = np.log(score)
      input_keys_and_scores.append((key, score))

    if input_tensor is None:
      input_tensor = torch.tensor([i[1] for i in input_keys_and_scores], dtype=torch.float64).reshape(-1, 1)
    else:
      input_tensor = torch.cat((input_tensor, torch.tensor([i[1] for i in input_keys_and_scores], dtype=torch.float64).reshape(-1, 1)), dim=1)
  
  output_tensor = model(input_tensor)

  with open(args.output_score_file, "w") as f_out:
    for i, s in enumerate(input_keys_and_scores):
      if args.label_column == 3:
        print(s[0], output_tensor[i].item(), file=f_out)
      elif args.label_column == 1:
        print(output_tensor[i].item(), s[0], file=f_out)
      else:
        raise Exception("Illegal label column: %d" % args.label_column)
