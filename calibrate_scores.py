#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.constant_(self.linear.weight, 1.0 / input_dim)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        out = self.linear(x)
        return out


def cllr(target_llrs, nontarget_llrs):
    """
    Calculate the CLLR of the scores
    """
    def negative_log_sigmoid(lodds):
        """-log(sigmoid(log_odds))"""
        return torch.log1p(torch.exp(-lodds))

    return 0.5 * (torch.mean(negative_log_sigmoid(target_llrs)) + torch.mean(negative_log_sigmoid(-nontarget_llrs)))/np.log(2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Calibrates speaker verification LLR scores")
  parser.add_argument('--save-model', help="Save calibration model to this file")
  parser.add_argument('--max-epochs', default=50)
  parser.add_argument('--label-column', metavar='label_column', type=int, default=3, choices=[1,3], help="label column")
  parser.add_argument('--log-llr', action='store_true', default=False, help="convert score to [0, 1] and log")
  parser.add_argument('key_file', help="Speaker recognition key file. Each line is a triple <enrolled_speaker> <test_speaker> target|nontarget")
  parser.add_argument('score_file', nargs='+', help="One or more score files. Each line is a triple <enrolled_speaker> <test_speaker> <LLR_score>")

  args = parser.parse_args()
  print(args)

  keys = {}
  for l in open(args.key_file):
    ss = l.split()
    if args.label_column == 3:
      trial = ss[0] + " " + ss[1]
      score = ss[2]
    elif args.label_column == 1:
      trial = ss[1] + " " + ss[2]
      score = ss[0]
    else:
        raise Exception("Illegal label column: %d" % args.label_column)
    if score == "tgt" or score == "target" or score == "1":
      is_target = True
    elif score == "imp" or score == "nontarget" or score == "0":
      is_target = False
    else:
      raise Exception("Illegal line in key file:\n%s" % l.strip())
    keys[trial] = is_target
  
  target_llrs = None
  nontarget_llrs = None
  for score_file in args.score_file:
    target_llrs_list = []
    nontarget_llrs_list = []

    for l in open(score_file):
      ss = l.split()
      if args.label_column == 3:
        trial = ss[0] + " " + ss[1]
        score = float(ss[2])
      elif args.label_column == 1:
        score = float(ss[0])
        trial = ss[1] + " " + ss[2]
      else:
        raise Exception("Illegal label column: %d" % args.label_column)
      if args.log_llr:
        score = (score + 1) / 2
        score = np.log(score)
      if keys[trial]:
        target_llrs_list.append(score)
      else:
        nontarget_llrs_list.append(score)

    if target_llrs is None:
      target_llrs = torch.tensor(target_llrs_list, dtype=torch.float64).reshape(-1, 1)
      nontarget_llrs = torch.tensor(nontarget_llrs_list, dtype=torch.float64).reshape(-1, 1)
    else:
      target_llrs = torch.cat((target_llrs, torch.tensor(target_llrs_list, dtype=torch.float64).reshape(-1, 1)), dim=1)
      nontarget_llrs = torch.cat((nontarget_llrs, torch.tensor(nontarget_llrs_list, dtype=torch.float64).reshape(-1, 1)), dim=1)

  start_cllr = cllr(target_llrs, nontarget_llrs)
  print("Starting point for CLLR is %f" % start_cllr)

  model = LinearModel(len(args.score_file))
  model.double()
  criterion = cllr

  optimizer = optim.LBFGS(model.parameters(), lr=0.01) 

  best_loss = 1000000.0

  for i in range(args.max_epochs):
    print('STEP: ', i)

    def closure():
        optimizer.zero_grad()
        new_nontarget_llrs = model(nontarget_llrs)
        new_target_llrs = model(target_llrs)
        loss = criterion(new_target_llrs, new_nontarget_llrs)
        print('  loss:', loss.item())
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    if (best_loss - loss < 1e-4):
      print("Converged!")
      break
    else:
      if loss < best_loss:
        best_loss = loss

  if args.save_model:
    print("Saving model to "+ args.save_model)
    torch.save(model.state_dict(), args.save_model)
