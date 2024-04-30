# Meeting notes
## 11 April:
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


## 23 april:
on incremental learning:
for each: point and line graph
- One with exchange rate
- one in "nice order"
- one in shuffled order 


## Notes after incremental learning experiment:


# Experiments:
## Transferability of knowledge (synthetic data)
- Datasets: even sine, odd sine, trends, combination (odd sine+even sine+trends), joker (sines of totally different frequencies, other trends)\

Part 1:
- train on even sine, try to predict on odd sine 
- Result: it outputs noise (=performs horribly). However, given one epoch training on odd sine, it performs well on both 

Part 2:
- train on even sine, odd sine, trends, joker
- test on combination, joker
- Result: it does not remember older datasets, just the last dataset that it saw is the only it performs on, combination does not work at all.

## Transferability of knowledge (real data)
- A, B datasets from same domain, such as temperature in different cities or ETTh1, ETTh2 etc (ETTh are hourly oil temperatures in electricity transformers)
- Train on A, both supervised and self-supervised, 
- test on A, test on B, measure performance, training time and epochs. We can later change A and B to other similar datasets.

**Results:** we find that on temperature, DLinear performs better than PatchTST. 
Furthermore, traditional PatchTST performs better than self-supervised.
On ETTh, PatchTST performs better than DLinear. Supervised training still outperforms self-supervised training. 
When it comes to different city temperatures, it boils down to what cities have more well behaving temperatures. 
For example, the models trained on Stockholm performs better out of the box on Malmö than they do on Stockholm

## Is learning additive or does it cause catastrophic forgetting?
For different similar datasets A, B (temperature in A: northern sweden, B: southern sweden or A: ETTh1 and B: ETTh2)
- Train on A, test on A, test on B (out of the box)
- for i in range(15)
  - train for one epoch on B
  - test on A and on B

**Hypothesis**: it will gradually forget the first dataset and therefore switch from performing well on A to performing well on B \
**Result**: it converges within one epoch and after that performs well on B. However it does not forget the original training. 
I believe this is because the structure of the data is too similar, they abilities do not compete 


## Is learning additive or does it cause catastrophic forgetting pt 2?
For different not-as-similar datasets A, B (A ETTh1, B temperature in Malmö)
- Train on A, test on A, test on B (out of the box)
- for i in range(15)
  - train for one epoch on B
  - test on A and on B

**Hypothesis**: this should surely make the model forget the first data right?\
**Result**: it again converges very fast and again, does not seem to forget too much. 
I believe this comes from some shared structures between the oil temperature in an electricity transformer and the weather (daily seasonality for example)
but it also indicates that the model is capable of performing on different datasets. 

We even cut the epochs so that they only train on a fraction of the data in order to force it to converge in a higher number of epochs, but it is still quite fast


## main experiment idea 
Topic: how much cross-dataset knowledge can be stored in a model of a certain size?
- with models: large patchTST, small patchTST, DLinear, (maybe more, quite scalable labour wise) 
- with datasets: A: temperature, B: ETTh, C: traffic, D: illness E: some kind of consumption F: (please share if you have any idea) 
- try to make a small "foundation"-model, namely one that can perform well on as many datasets as possible. 
  - Train on A, test on A-F
  - Train on A, B, test on A-F
  - and so on.
- Try different strategies for training 
  - one after the other,
  - sampling from all during training, 
  - "dreaming"/replay buffer

Hypothesis: the datasets that share many seasonality components such as traffic and temperature will not compete about the same capacity in the model.
What will be interesting is if the model can keep track of datasets with different seasonality. We are therefore looking for datasets from other domains.


# Other
## Thoughts:
- The positional encoding is an additive matrix of dimension DxN, meaning each dimension in the latent space gets an assigned number for each patch. Does this not mean that the time granularity for keeping track of frequencies is reduced to ```patch_len```?
- How to train on several datasets without overwriting previous knowledge? Train on all simultaneously :)
- Casual self-attention? 


# Todo:
- redo incremental in other order
- Populate slides after feedback
- redo multi-learn experiment with: 
  - normalised loss
  - also self-supervised 
  - target specific test logging
- Go over thesis
  - Go over Amin's comments
- Write about experiments we have done
  - Multi learning:
    - Setup done
    - results need to redo
    - comment
- Write problem definition (create a function that maps x to x)

- implement dreaming
- test dreaming
- implement normalised loss 
- Make staddle model :)

- more data in order to keep benchmark data
- find a good schema for learning rate when doing pretraining/finetuning

### Finished:- 
Finish slides for presentation
  - Send to Amin and Rahul
- Incremental learning:
  - Setup done
  - results done
  - comments done
- Write about datasets
- Write about Linear models
- Write about previous transformer based models
- rewrite plotting to fit multi-learn
- - test multi-learn: (done-ish)
- for model in:
  - Patch_small, 
  - Patch_norm, 
  - Patch_large, 
  - Patch_small_self, 
  - Patch_norm_self, 
  - Patch_large_self,
  - Linear, 
  - NLinear, 
  - DLinear, 
  - Naive, 
  - daily repeater, 
  - full repeater
- Train on multidata, test on components
- compare with expert_models

- Incremental learning experiment: (done-ish)
  - datasets = [traffic_small('2'), temp_small, ETTh1, Illness, Exchange]
  - train an expert model for each dataset, save MSE and NRV
  - for model in [Patch_small, Patch_norm, Patch_large, Linear, NLinear, DLinear, Naive, daily repeater, full repeater]
    - learning rate <- 10^-4
    - for dataset in datasets:
      - train model on dataset
      - test on all datasets
      - learning rate <- 10^-5
    - test out of the box on unseen data: ETTh2, other temperature, other traffic
- Analyse incremental learning results
- if time: adapt everything to always use Multidataset
- when switching datasets, validate first so that validation loss is compared to out of box performance of new dataset
- Add inherited loss to finetuning
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

- study how to train on several training sets:
  - timeGPT
  - A decoder only foundation model
  - foundation models in NLP
- Results:
  - it seems to be on the topic of continual learning, incremental learning, overcoming catastrophic forgetting:



  


 