## Thoughts:
- The positional encoding is an additive matrix of dimension DxN, meaning each dimension in the latent space gets an assigned number for each patch. Does this not mean that the time granularity for keeping track of frequencies is reduced to ```patch_len```?
- How to train on several datasets without overwriting previous knowledge?

## Notes for meeting 11 April:
- need to update experiment *see experiment*
- according to plan we have small presentation at SEB next week *we will decide on a date when we have some preliminary results to discuss*



## Experiment:
A, B, C are different cities
Train on A, both supervised and self-supervised, 
test on A, test on B, measure performance, training time and epochs

We can later change A and B to other similar datasets



## Todo:
- Analyse effect of pretraining etc
- Fix benchmark-models:
  - DLinear. done
  - Naive, just predict the last value forward. done 
  - "smarter" naive: normalise input using first value in sequence, take last value in sequence and add normalised input. Done
- Gather datasets:
  - A: Train: Stockholm done
  - B: Söderhamn, (Delsbo instead): done
  - C: Sundsvall: done
  - D: Jönköping: done
  - E: Lund (Malmö instead) done

- Before modelling!! move old metrics.csv

for setting in [supervised, self-supervised]: 
- train on A
- test on A-E


  


 