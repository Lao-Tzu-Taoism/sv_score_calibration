The VOiCES from a Distance Challenge 2019 scorer package (v1.0)

-------------------
About:
This package can be used for two purposes:
1) to evaluate system performance of scores in an ascii file given a trial key, or
2) to check that a scorefile satisfies the requirement for submission based on a trial list

The scoring process provide four metrics; the primary metric actual Cdet (actDCF), 
and alternate metric log likelihood ratio cost (Cllr), then other common measures
including minimum Cdet (minDCF), equal error rate (EER), and average R-precision (avgRPrec) [Brummer06].

The submission checking process runs a number of check on a scorefile given a
trial list to ensure it satisfies the requirements for submission.

More details with example for each process are listed below.


-------------------
File definitions and examples:
scorefile: 3-column ascii file with [modelID testID score]
(i.e., system output example:
Lab41-SRI-VOiCES-rm1-none-sp0083-ch003054-sg0005-mc01-stu-clo-dg090 sid_dev/Lab41-SRI-VOiCES-rm2-babb-sp0032-ch021625-sg0007-mc02-lav-clo-dg120.wav -31.60794
Lab41-SRI-VOiCES-rm1-none-sp0083-ch003054-sg0005-mc01-stu-clo-dg090 sid_dev/Lab41-SRI-VOiCES-rm2-babb-sp0032-ch021625-sg0007-mc04-lav-mid-dg120.wav -30.79861
Lab41-SRI-VOiCES-rm1-none-sp0083-ch003054-sg0005-mc01-stu-clo-dg090 sid_dev/Lab41-SRI-VOiCES-rm2-babb-sp0032-ch021625-sg0007-mc06-lav-far-dg120.wav -44.45553
)

keyfile:   3-column ascii file with [modelID testID tgt/imp]
(i.e., ./dev/keys/core-core.lst
Lab41-SRI-VOiCES-rm1-none-sp3446-ch144019-sg0006-mc03-stu-mid-dg080 sid_dev/sp3446/Lab41-SRI-VOiCES-rm2-musi-sp3446-ch144021-sg0018-mc08-lav-beh-dg090.wav tgt
Lab41-SRI-VOiCES-rm1-none-sp3446-ch144019-sg0006-mc03-stu-mid-dg080 sid_dev/sp3521/Lab41-SRI-VOiCES-rm2-musi-sp3521-ch012715-sg0017-mc12-lav-wal-dg090.wav imp
Lab41-SRI-VOiCES-rm1-none-sp3446-ch144019-sg0006-mc03-stu-mid-dg080 sid_dev/sp3521/Lab41-SRI-VOiCES-rm2-musi-sp3521-ch012715-sg0006-mc04-lav-mid-dg120.wav imp
)

triallist: 2-column ascii file with [modelID testID]
(i.e., ./dev/lists/trial-core-core.lst
Lab41-SRI-VOiCES-rm1-none-sp3446-ch144019-sg0006-mc03-stu-mid-dg080 sid_dev/sp3521/Lab41-SRI-VOiCES-rm2-musi-sp3521-ch012715-sg0017-mc11-lav-ceo-dg090.wav
Lab41-SRI-VOiCES-rm1-none-sp3446-ch144019-sg0006-mc03-stu-mid-dg080 sid_dev/sp3521/Lab41-SRI-VOiCES-rm2-musi-sp3521-ch012715-sg0017-mc04-lav-mid-dg090.wav
Lab41-SRI-VOiCES-rm1-none-sp3446-ch144019-sg0006-mc03-stu-mid-dg080 sid_dev/sp3521/Lab41-SRI-VOiCES-rm2-musi-sp3521-ch012715-sg0006-mc10-lav-cec-dg120.wav
)


-------------------
Scoring Usage:
$ score_voices <scorefile> <keyfile>

Scoring Example:
$ score_voices scorefile dev-trial-keys.lst
minDCF   : 0.9170
actDCF   : 0.9468
avgRPrec : 0.4435
EER      : 0.1691
Cllr     : 0.8441


-------------------
Submission Check Usage:
$ score_voices -c <scorefile> <triallist>

Submission Check Example:
$ score_voices -c scorefile dev-trial.lst
Submission check PASSED

Submission Checks:
1) All trials from triallist exist in scorefile
2) No trial duplicates exist in scorefile
3) No additional trials are in the scorefile
4) The format of model/test labels are correct (i.e., flac extension on test segments)
5) No scores are NaN.
