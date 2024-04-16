## Thoughts:
- The positional encoding is an additive matrix of dimension DxN, meaning each dimension in the latent space gets an assigned number for each patch. Does this not mean that the time granularity for keeping track of frequencies is reduced to ```patch_len```?
- How to train on several datasets without overwriting previous knowledge?

## Notes for meeting 11 April:
- need to update experiment *see experiment*
- according to plan we have small presentation at SEB next week *we will decide on a date when we have some preliminary results to discuss*

## Notes 16 april:
A: Malmo B: Katterjakk
- take one method for learning
- Train fully on A
- Test on A: Ma
- Test out of box on B
- Train one epoch on B 
- Test on A & B
- Train one more epoch on B
- Test on A & B
- If performance is completely flipped:
  - Decrease LR

- Then, after finding that we (probably) forget how to behave in one domain when learning the other, we come up with ways to mitigate this:
  - In-epoch mixing
  - Between epoch mixing


## Experiment:
A, B, C are different cities
Train on A, both supervised and self-supervised, 
test on A, test on B, measure performance, training time and epochs. We can later change A and B to other similar datasets.
**Results:** we find that on temperature, DLinear performs better than PatchTST. Furthermore, traditional PatchTST performs better than self-supervised.
On ETTh, PatchTST performs better than DLinear. Supervised training still outperforms self-supervised training. 





## Todo:
- study how to train on several training sets:
  - timeGPT
  - A decoder only foundation model
  - foundation models in NLP
- Results:
  - it seems to be on the topic of continual learning, incremental learning, overcoming catastrophic forgetting:
  - 


- Add inherited loss to finetuning
- find a good schema for learning rate when doing pretraining/finetuning



### Finished:
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

- Before experiment!! move old metrics.csv

for setting in [supervised, self-supervised]: 
- train on A
- test on A-E

- Same but using ETTh1 and ETTh2




  


 