# Score calibrator for speaker verification

This software allows you to calibrate log likelihood ratio (LLR) scores 
for speaker verification evaluation.

Often, speaker verification systems are evaluated on the actual DCF scores
or CLLR scores. In order to get good performance on these measures, LLR
scores from a speaker verification system (e.g., from a PLDA model) need
to be calibrated.

This software allows one to optimize the CLLR measure of a speaker 
verification system (or a combination of systems). 
Calibration is done by finding a linear transform
that optimizes the CLLR measure of the heldout data. 

## Requirements

  * Python 3+
  * Pytorch 1.0 (no GPU needed)

## Example

In the `samples`  directory, there are thee files: `sys1_llr.txt`,
`sys2_llr.txt` and `trial-keys.txt`. The first two are (uncalibrated)
LLR scores from two different systems on a heldout trial set, and the third file
gives the oracle values for all trials -- target (`tgt`) or non-target/impostor (`imp`).

In order to measure the DCF and CLLR scores, you need to download and extract a scoring tool
from https://app.box.com/s/9tpuuycgxk9hykr6romsv05vvmdpie11/file/389271165078.
It's an official scorer for the [The VOiCES from a Distance Challenge](https://voices18.github.io/Interspeech2019_SpecialSession/).
[The VOiCES from a Distance Challenge 2019 Evaluation Plan](https://arxiv.org/pdf/1902.10828.pdf)

Let's first measure the accuracy of the uncalibrated scores:

    $ python2 voices_scorer/score_voices sample/sys1_llr.txt sample/trial-keys.txt                 
    minDCF   : 0.4252
    actDCF   : 0.7496
    avgRPrec : 0.6384
    EER      : 0.0547
    Cllr     : 0.9787

    $ python2 voices_scorer/score_voices sample/sys2_llr.txt sample/trial-keys.txt                 
    minDCF   : 0.3034
    actDCF   : 1.8849
    avgRPrec : 0.6979
    EER      : 0.0710
    Cllr     : 0.5986

The software can calibrate scores of one or more systems. Let's first try
to calibrate the 1st system (`sample/sys2_llr.txt`). First, you have to find
the parameters that optimize CLLR of heldout data:

    $ python calibrate_scores.py --save-model sample/sys1_calibration.pth sample/trial-keys.txt sample/sys1_llr.txt                                                           
    Starting point for CLLR is 0.978737
    STEP:  0
      loss: 0.5246010594024472
      [...]
      loss: 0.18544731777635964
    Converged!
    Saving model to sample/sys1_calibration.pth
    
Next, you need to *apply* the calibration model:

    $ python apply_calibration.py sample/sys1_calibration.pth sample/sys1_llr.txt sample/sys1_calibrated_llr.txt
    
Let's measure the performance of the calibrated system:

    $ python2 voices_scorer/score_voices sample/sys1_calibrated_llr.txt sample/trial-keys.txt      
    minDCF   : 0.4252
    actDCF   : 0.4320
    avgRPrec : 0.6384
    EER      : 0.0547
    Cllr     : 0.1854

As can be seen, the `actDCF` and `Cllr` scores are now much better than initially.

You can also calibrate a fusion of two or more systems:

    $ python calibrate_scores.py --save-model sample/sys1_sys2_calibration.pth sample/trial-keys.txt sample/sys1_llr.txt sample/sys2_llr.txt                                 
    Starting point for CLLR is 0.788658
    STEP:  0
      loss: 0.711224191738577
      loss: 0.7045015511238044
      [...]
      loss: 0.18383203478911536
      loss: 0.18382984498508542
    Converged!
    Saving model to sample/sys1_sys2_calibration.pth

Apply the model:

    $ python apply_calibration.py sample/sys1_sys2_calibration.pth sample/sys1_llr.txt sample/sys2_llr.txt sample/sys1_sys2_calibrated_llr.txt 
    
Measure the performance:

    $ python2 voices_scorer/score_voices sample/sys1_sys2_calibrated_llr.txt sample/trial-keys.txt 
     
    minDCF   : 0.3516
    actDCF   : 0.3586
    avgRPrec : 0.6592
    EER      : 0.0533
    Cllr     : 0.1838


You can cite the following paper if you use the software in research:

    @inproceedings{alumae2019taltech,
      author={Tanel Alum\"{a}e, Asadullah},
      title={The {TalTech} Systems for the {VOiCES from a Distance Challenge}},
      year=2019,
      booktitle={Interspeech (submitted)},
    }

## Log-likelihood ratio(LLR)
The speaker recognition system is required to process each trial independently and output a log-likelihood ratio (LLR), using natural (base e) logarithm, for that trial. The LLR for a given trial including a test
segment s is defined as follows:

`LLR(s) = log(p(s|H0)/p(s|H1))` 

where P(.) denotes the probability distribution function
(pdf), and H0 and H1 represents the null (i.e., s is spoken by the
enrollment speaker) and alternative (i.e., s is not spoken by the
enrollment speaker) hypotheses, respectively. The performance
of a speaker recognition system will be judged on the accuracy
of these LLRs.

I think when using cosine backend, LLR is `log(cosine score)`. Or, just the log score of (enroll, test), i.e. `log(score(enroll, test))`. 

```
LLR = log(cosine score) = log(s)
neg_log_sigmoid(lodds) = torch.log1p(torch.exp(-lodds) = -log(sigmoid(lodds))
Cllr = 1/(2xlog(2)) * ( sum(log(1+1/s)) / Ntar + sum(log(1+s)) / Nnon)
     =  0.5 * (mean(log(1+1/s_tar)) + mean(log(1+s_non))) / log(2)
     = 0.5 * (mean(neg_log_sigmoid(s_tar_llr)) + mean(neg_log_sigmoid(-s_non_llr))) / log(2)
```
## Refs

### Calibration
* http://www.cs.joensuu.fi/pages/franti/sipu/pub/Score_calibration_Mandasari_2014.pdf
* https://arxiv.org/pdf/2002.03802.pdf
* https://www.kth.se/polopoly_fs/1.731834.1550156159!/HenrikBostrom_Slides.pdf
* https://medium.com/@kingsubham27/calibration-techniques-and-its-importance-in-machine-learning-71bec997b661

* https://github.com/fabiankueppers/calibration-framework.git
* https://github.com/SubhamIO/Calibration-Techniques-in-Machine-Learning
* https://github.com/donlnz/nonconformist
