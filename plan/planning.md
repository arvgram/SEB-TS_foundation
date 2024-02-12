
    date created: 10 Feb 2024
    last update: 12 Feb 2024
    
## Phase (I) (-- March 15)
- Be able to generate synthetic timeseries with **Seasonal**, **Trend**, **Random Walk**, **Noise** components. 
    - To control the form of generated timeseries, the ratio of coefficients for each component is specified.
- Understand `Patch-tst` implementation:
    - Patching mechanism: size of patch and the overlap, 
    - Supervised vs. self-supervised approaches
    - Any form of feature engineering or external features added to the raw data : (i) Fourier terms (ii) human-related time interval (day, week, ...)
- One example of transfer learning: Generate N dataset with a change of only one of the timeseries parameter (s,t,r) Train the model on (N-1) and leave one out to measure the capacity of the model to capture relations of between segments in unseen data. (a simple experiment to explore the idea of foundation model)
- Understand `Linear layer` model from critical paper.
- Have a sense of training time for different architecture specification. (ts size, patch size and overlap, number of epochs)
- Maybe a quick experiment with TimeGPT API>

## Phase (II) (15 March -- 15 April)
- Working on the first draft of thesis.