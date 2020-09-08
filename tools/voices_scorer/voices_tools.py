#!/usr/bin/env python

"""
Helper functions for scoring VOiCES scorefiles or checking submissions

Author: Mitchell McLaren
Email: mitchell.mclaren@sri.com
Date: January 7. 2016 (v2 SITW scoring script)
Date: January 23, 2019 (revised for VOiCES)
"""

import sys,os
import numpy as np
import gzip

def ismember(a, b):
    """ A replica of the MATLAB function """
    b = dict((e, i) for i, e in enumerate(b))
    rindex = np.array([b.get(k, -1) for k in a])
    return rindex != -1, rindex

def logit(p):
    """Logit function"""
    p = np.atleast_1d(np.asfarray(p))
    logit_p = np.zeros_like(p)
    valid = (p > 0) & (p < 1)
    if np.any(valid):
        logit_p[valid] = np.log(p[valid] / (1 - p[valid]))
    logit_p[p==0] = np.NINF
    logit_p[p==1] = np.PINF
    return logit_p

class Key(object):
    """
    A Key object to filter tar/non scores from a scorefile
    The object is in full-matrix form ie: it contains
    train_ids: list of string of modelids
    test_ids: list of string of test_ids
    mask: a len(train_ids)xlen(test_ids) int matrix containing
        +1: for target trial
        -1: for impostor trial
        0: for non scored trial
    """
    def __init__(self, trainids, testids, mask, name="none"):
        self.train_ids=list(trainids)
        self.test_ids=list(testids)
        if(mask.shape!=(len(trainids),len(testids))):
            raise Exception("Incompatible mask for creation of key")
        self.mask=mask.astype(np.int8)
        self.name=name

    @classmethod
    def load(self, filename, notruth=False, label_column=3):
        """ Build Key from a text file with the following format
        trainID testID tgt/imp
        If notruth is True, then load a trial list for submission checking:
        trainID testID
        """
        func = gzip.open if os.path.splitext(filename)[1] == ".gz" else open
        with func(filename) as f:
            L = [line.strip().split(' ')[0:3] for line in f]
        L.sort()
        if label_column == 3:
            if notruth:
                try:
                    trainids, testids = [list(set(t)) for t in zip(*L)]
                except ValueError as e:
                    raise Exception("Need 2 columns for a trial list. Did you pass a key instead? If scoring a file with a key, don't use -c on score_voices.")
            else:
                try:
                    trainids, testids, _ = [list(set(t)) for t in zip(*L)]
                except ValueError as e:
                    raise Exception("Need 3 columns for a key. Did you pass a trial list? If checking a submission, use score_voices -c.")
        elif label_column == 1:
            if notruth:
                try:
                    trainids, testids = [list(set(t)) for t in zip(*L)]
                except ValueError as e:
                    raise Exception("Need 2 columns for a trial list. Did you pass a key instead? If scoring a file with a key, don't use -c on score_voices.")
            else:
                try:
                    _, trainids, testids = [list(set(t)) for t in zip(*L)]
                except ValueError as e:
                    raise Exception("Need 3 columns for a key. Did you pass a trial list? If checking a submission, use score_voices -c.")
        else:
            raise Exception(f"not support label column: {label_column}")
        idxtrainids = dict([(x,i) for i,x in enumerate(trainids)])
        idxtestids = dict([(x,i) for i,x in enumerate(testids)])
        nbT, nbt = len(trainids), len(testids)

        # default mask to have all trials not included ('0')
        mask = np.zeros((nbT, nbt),dtype=np.int8)
        for trial in L:
            if label_column == 3:
                if notruth: # For submission checking
                    mask[idxtrainids[trial[0]], idxtestids[trial[1]]] = 2
                else:
                    if trial[2] == "tgt" or trial[2] == 'target' or trial[2] == '1':
                        mask[idxtrainids[trial[0]], idxtestids[trial[1]]] = 1
                    if trial[2] == "imp" or trial[2] == 'nontarget' or trial[2] == '0':
                        mask[idxtrainids[trial[0]], idxtestids[trial[1]]] = -1
            elif label_column == 1:
                if notruth: # For submission checking
                    mask[idxtrainids[trial[1]], idxtestids[trial[2]]] = 2
                else:
                    if trial[0] == "tgt" or trial[0] == 'target' or trial[0] == '1':
                        mask[idxtrainids[trial[1]], idxtestids[trial[2]]] = 1
                    if trial[0] == "imp" or trial[0] == 'nontarget' or trial[0] == '0':
                        mask[idxtrainids[trial[1]], idxtestids[trial[2]]] = -1
            else:
                raise Exception(f"not support label column: {label_column}")
        return Key(trainids, testids, mask, os.path.basename(filename))


class Scores(object):
    """
    A Scores object to hold trial scores from a scorefile
    The object is in full-matrix form ie: it contains
    train_ids: list of string of modelids
    test_ids: list of string of test_ids
    score_mat: a len(train_ids)xlen(test_ids) float matrix containing all scores
    mask: a len(train_ids)xlen(test_ids) int matrix containing
        1: for included trial
        0: for non scored trial
    """
    def __init__(self, trainids, testids, score_mat, score_mask, name="none"):
        self.train_ids=list(trainids)
        self.test_ids=list(testids)
        if (score_mask.shape!=(len(trainids),len(testids))) or (score_mat.shape!=(len(trainids),len(testids))):
            raise Exception("Incompatible score_mask or score_mat size for creation of Scores")
        self.score_mask=score_mask.astype(np.int8)
        self.score_mat=score_mat.astype('f')
        self.name=name

    @classmethod
    def load(cls, filename, label_column=3):
        """ Build Scores from a text file with the following format
        trainID testID score
        filename can be ascii or gzip ascii with .gz extension
        """
        func = gzip.open if os.path.splitext(filename)[1] == ".gz" else open
        with func(filename) as f:
            L = [line.strip().split(' ')[0:3] for line in f]
        L.sort()
        if label_column == 3:
            trainids, testids, _ = [list(set(t)) for t in zip(*L)]
        elif label_column == 1:
            _, trainids, testids = [list(set(t)) for t in zip(*L)]
        else:
            raise Exception(f"not support label column: {label_column}")
        idxtrainids = dict([(x,i) for i,x in enumerate(trainids)])
        idxtestids = dict([(x,i) for i,x in enumerate(testids)])
        nbT, nbt = len(trainids), len(testids)

        # default is to not include score
        score_mask = np.zeros((nbT, nbt),dtype=np.int8)
        score_mat = np.NINF * np.ones((nbT, nbt),dtype='f')
        for trial in L:
            if label_column == 3:
                score_mask[idxtrainids[trial[0]], idxtestids[trial[1]]] = 1
                if score_mat[idxtrainids[trial[0]], idxtestids[trial[1]]] == np.NINF:
                    score_mat[idxtrainids[trial[0]], idxtestids[trial[1]]] = trial[2]
                else:
                    # Being re-defined!
                    raise Exception("Trial [%s %s] appears more than once in the score file!" % (trial[0],trial[1]))
            elif label_column == 1:
                score_mask[idxtrainids[trial[1]], idxtestids[trial[2]]] = 1
                if score_mat[idxtrainids[trial[1]], idxtestids[trial[2]]] == np.NINF:
                    score_mat[idxtrainids[trial[1]], idxtestids[trial[2]]] = trial[0]
                else:
                    # Being re-defined!
                    raise Exception("Trial [%s %s] appears more than once in the score file!" % (trial[1],trial[2]))
            else:
                raise Exception(f"not support label column: {label_column}")
        return Scores(trainids, testids, score_mat, score_mask, os.path.basename(filename))


    def align(self, key):
        """
        Will align a Scores object with a key
        This will create a new scorefile, aligned with the keyfile
        Mask is aligned with the key, with zeros for missing trials
        @param key: loaded Key object
        """
        # map from the index in the current scorefile to the key file
        nb_models = len(key.train_ids)
        nb_tests = len(key.test_ids)
        (hasmodel, rindx) = ismember(key.train_ids, self.train_ids)
        (hasseg, cindx) = ismember(key.test_ids, self.test_ids)
        rindx = rindx[hasmodel]
        cindx = cindx[hasseg]

        # CHECKS to confirm all trials are represented
        if(np.sum(hasmodel) < nb_models):
            if np.sum(hasmodel)==0:
                print("#####################")
                print("No valid models found in score file! Perhaps you passed the wrong score file or the model format is incorrect. The key expected sometime like:")
                print([x for x in sorted(key.train_ids)[:5]])
                print("and your scorefile contains (e.g.):")
                print([x for x in sorted(self.train_ids)[:5]])
                print("#####################")
                raise Exception("No valid models in score file!")
            else:
                raise Exception("Missing %d models from your score file. Found %d but expected %d." \
                    % (nb_models-np.sum(hasmodel),np.sum(hasmodel),nb_models))

        if(np.sum(hasseg) < nb_tests):
            if np.sum(hasseg)==0:
                print("#####################")
                print("No valid test segments found in score file! Perhaps you passed the wrong score file or the test format is incorrect. The key expected sometime like:")
                print([x for x in sorted(key.test_ids)[:5]])
                print("and your scorefile contains (e.g.):")
                print([x for x in sorted(self.test_ids)[:5]])
                print("#####################")
                raise Exception("No valid test segments in score file!")
            else:
                raise Exception("Missing %d test segments from your score file. Found %d but expected %d." \
                    % (nb_tests-np.sum(hasseg),np.sum(hasseg),nb_tests))


        # New scorefile
        new_score_mat = np.NINF * np.ones(key.mask.shape, 'f')
        new_score_mask = np.zeros_like(key.mask)
        # The use of ix_ enables indexing with row / column indexes
        # Different from linear indexing with arrays
        new_score_mat[np.ix_(hasmodel, hasseg)] = self.score_mat[np.ix_(rindx, cindx)]
        # Wherever key.mask is 0, turn corresponding score to NINF (even if it is defined in original score_mat)
        new_score_mat[key.mask==0] = np.NINF
        new_score_mask[np.ix_(hasmodel, hasseg)] = self.score_mask[np.ix_(rindx, cindx)]
        new_score_mask[key.mask==0] = 0

        # key.mask.max()!=2 indicates a truth key
        if (key.mask.max()!=2 and ((key.mask!=0).sum() != (new_score_mask!=0).sum())):
            raise Exception("Missing %d trials in score file!" % ((key.mask!=0).sum()-(new_score_mask!=0).sum()))
        # key.mask.max()!=2 indicates a trial list for submission checking
        if (key.mask.max()==2 and ((key.mask!=0).sum() != (self.score_mask!=0).sum())):
            raise Exception("Incorrect number of trials in score file! Found %d trials but expected %d." % ((self.score_mask!=0).sum(),(key.mask==2).sum()))
        # Check for NaN's
        if (np.any(~np.isfinite(new_score_mat[new_score_mask==1]))):
            raise Exception("Found %d NaN's in score file!" % (((~np.isfinite(new_score_mat[new_score_mask==1])).sum())))

        return Scores(key.train_ids, key.test_ids, new_score_mat, new_score_mask, name=self.name)



def print_performance(scores,key,p_tar=0.01):
    """
    Print to screen the average R-Precision over enrolled models
    """
    ascores = scores.align(key)
    Pfa,Pmiss,sortedscores,tar,non = det(ascores,key)
    print("minDCF   : %.4f" % (get_min_dcf(Pfa, Pmiss, p_tar=p_tar)))
    print("actDCF   : %.4f" % (get_act_dcf(Pfa, Pmiss, sortedscores, p_tar=p_tar)))
    print("avgRPrec : %.4f" % (get_avg_rprecision(scores,key)))
    print("EER      : %.4f" % (get_eer(Pfa, Pmiss)))
    print("Cllr     : %.4f" % (get_cllr(tar,non)))


def get_eer(Pfa,Pmiss):
    """
    Calculate EER
    """
    idxeer=np.argmin(np.abs(Pfa-Pmiss))
    return 0.5*(Pfa[idxeer]+Pmiss[idxeer])


def get_avg_rprecision(scores,key):
    """ Find Average R-Precision for information retrieval"""
    ascores = scores.align(key)
    # Find num target trials per model
    tgtcount = np.sum(key.mask==1,axis=1)
    # Sort socre matrix for rapid indexing
    srtidx = np.argsort(scores.score_mat)
    rprec = []
    for i,cnt in enumerate(tgtcount):
        if cnt==0:
            # SKIP
            continue

        # Get array of tgt/imp for highest cnt scores
        srttgtnon = key.mask[i,srtidx[i,:]]
        cutidx = np.where(np.abs(srttgtnon[::-1]).cumsum()==cnt)[0][0]
        srttgtnon = srttgtnon[-cutidx-1:]
        # Calculate precision
        rprec.append((srttgtnon==1).sum()/float(cnt))
     
    avgrprec = np.mean(rprec)
    return avgrprec

def get_act_dcf(Pfa, Pmiss, sorted_scores, p_tar=0.01, normalize=True):
    """
    input:
        p_tar: a vector of target priors
        normalize: normalize DCFs
    output:
        Values of actualDCF, for optimal thresholds assuming scores are LLR
    """
    p_tar = np.asarray(p_tar)
    p_non = 1 - p_tar
    plo = -1. * logit(p_tar)
    dcfs, idxdcfs = np.zeros_like(p_tar), np.zeros_like(p_tar)

    idx = sorted_scores.searchsorted(plo)
    dcfs = p_tar * Pmiss[idx] + p_non * Pfa[idx]
    idxdcfs = idx
    if normalize:
        mins = np.amin(np.vstack((p_tar,p_non)),axis=0)
        dcfs /= mins

    return dcfs.squeeze()

def get_min_dcf(Pfa, Pmiss, p_tar=0.01, normalize=True):
    """
    input:
        p_tar: a vector of target priors
        normalize: normalize DCFs
    output:
        Values of minDCF, one for each value of p_tar
    """
    p_tar = np.asarray(p_tar)
    p_non = 1 - p_tar
    # CDet = CMiss x PTarget x PMiss|Target + CFalseAlarm x (1-PTarget) x PFalseAlarm|NonTarget
    cdet = np.dot(np.vstack((p_tar, p_non)).T, np.vstack((Pmiss,Pfa)))
    idxdcfs = np.argmin(cdet, 1)
    dcfs = cdet[np.arange(len(idxdcfs)), idxdcfs]

    if normalize:
        mins = np.amin(np.vstack((p_tar, p_non)), axis=0)
        dcfs /= mins
    return dcfs.squeeze()

def det(scores,key):
    """
    Retrieves the Pfa and Pmiss vectors and a sorted score vector
    These are used for EER, and DCF calculation
    """
    tar    = scores.score_mat[key.mask==1]
    non    = scores.score_mat[key.mask==-1]
    ntrue  = tar.shape[0]
    nfalse = non.shape[0]
    ntotal = ntrue+nfalse
    if(ntrue==0):
        raise Exception("No target trials found")
    if(nfalse==0):
        raise Exception("No impostor trials found")

    Pmiss  = np.zeros(ntotal+1,np.float32) # 1 more for the boundaries
    Pfa    = np.zeros_like(Pmiss)

    ascores = np.zeros((ntotal,2),np.float32)
    ascores[0:nfalse,0]=non
    ascores[0:nfalse,1]=0
    ascores[nfalse:ntotal,0]=tar
    ascores[nfalse:ntotal,1]=1

    ## Sort DET scores.
    # Scores are sorted along the first row while keeping the second row fix (same as MATLAB sortrows)
    ascores = ascores[ascores[:,0].argsort(),]

    sumtrue = np.cumsum(ascores[:,1])
    sumfalse = nfalse - (np.arange(1,ntotal+1)-sumtrue)

    miss_norm = ntrue
    fa_norm = nfalse

    Pmiss[0]  = float(miss_norm-ntrue) / miss_norm
    Pfa[0]    = float(nfalse) / fa_norm
    Pmiss[1:] = (sumtrue+miss_norm-ntrue) / miss_norm
    Pfa[1:]   = sumfalse / fa_norm

    return Pfa,Pmiss,ascores[:, 0],tar,non

def get_cllr(tar_llrs,non_llrs):
    """
    Calculate the CLLR of the scores
    """
    def neglogsigmoid(lodds):
        """-log(sigmoid(log_odds))"""
        return np.log1p(np.exp(-lodds.astype(np.float64)))

    return (np.mean(neglogsigmoid(tar_llrs)) + np.mean(neglogsigmoid(-non_llrs)))/2/np.log(2)
