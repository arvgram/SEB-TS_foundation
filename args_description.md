### `args`-explanations
#### Setup:
- `random_seed`: seed for all randomized processes
- `is_training`: I don't think it is used.
- `root_path`: the path from current location to dataset directory
- `data_file_name`: name of data file
- `model_id`: a name for model
- `model`: whether to use PatchTST, Autoformer, Transformer, Informer, DLinear, NLinear, Linear
- `data`: if to use one of the standard sets (ETTh1 etc) or custom
- `features`: forecasting task, M: multivariate (mv) predict mv, MS: mv predict univariate (uv), S: uv predict uv
- `target`: name of column to predict (for S and MS)
- `checkpoints`: where to store model checkpoints
- `do_predict`: if you want to predict on new data. 
- `output_attention`: if enabled, the models output the attention for access (in former models only).

#### Modelling:
**Task**:
- `freq`: [s,t,h,d,b,w,m], frequency for feature encoding. Not sure where it is used
- `seq_len`: L, lookback window
- `label_len`: start token length... It concatenates label_len number of zeros to encoder input? I do not really understand
- `pred_len`: T, prediction horizon, how many periods forward

**Regularisation**:
- `fc_dropout`: regularisation of fully connected layers by switching off nodes with prob fc_dropout
- `head_dropout`: regularisation of head

**PatchTST specific**:
- `patch_len`: P, the length of the patches, the tokens
- `stride`: S, the non-overlapping region between two consecutive patches
- `padding_patch`: repeat the last number of the input sequence S number of times
- `revin`: REVin, Reversable instance normalization, essentially subtracting mean and dividing by variance for each channel in lookback: $$ \hat{x}_{k t}^{(i)}=\left(\frac{x_{k t}^{(i)}-\mathbb{E}_t\left[x_{k t}^{(i)}\right]}{\sqrt{\mathbb{V}\left[x_{k t}^{(i)}\right]+\epsilon}}\right)$$ and then adding it after prediction: $$ \hat{y}_{k t}^{(i)}=\sqrt{\mathbb{V}\left[x_{k t}^{(i)}\right]+\epsilon} \cdot\left(\tilde{y}_{k t}^{(i)}\right)+\mathbb{E}_t\left[x_{k t}^{(i)}\right] $$
- `affine`: for REVin, includes trainable parameters that allows flexibility in the location/scale shift: $$ \hat{x}_{k t}^{(i)}=\gamma_k\left(\frac{x_{k t}^{(i)}-\mathbb{E}_t\left[x_{k t}^{(i)}\right]}{\sqrt{\mathbb{V}\left[x_{k t}^{(i)}\right]+\epsilon}}\right)+\beta_k $$ which renders output:$$ \hat{y}_{k t}^{(i)}=\sqrt{\mathbb{V}\left[x_{k t}^{(i)}\right]+\epsilon} \cdot\left(\frac{\tilde{y}_{k t}^{(i)}-\beta_k}{\gamma_k}\right)+\mathbb{E}_t\left[x_{k t}^{(i)}\right] $$
- `subtract_last`: using last value in lookback for normalisation instead of mean in revIN. It is not mentioned in revIN paper.
- `decomposition`: if enabled, the PatchTST-model object consists of two different PatchTST_backbone modules and a `series_decompostion`. The `series_decompostion` object returns a moving average and a residual, and these are sent to different backbones (`model_trend` and `model_res`).
- `kernel_size`: is passed to PyTorch `AvgPool1D`, specifies the length of the moving average in decomposition (if it is enabled)
- `individual`: if to use individual head or not. I don't really understand.

**Generic model arguments**
- `embed_type`: The type of embedding of input data to be used in all of the -former models in the paper. 0: default, 1: val+temp+pos 2: val+temp, 3: val+pos, 4: val
- `enc_in`: Encoder input size. Transferred to `c_in` and `n_vars` in PatchTST.
- `dec_in`: Decoder input size. Seems to only be used in -former models.
- `c_out`: Output dimension
- `d_model`: latent space dimension
- `n_heads`: number of attention heads
- `e_layers`: number of encoder layers, is converted to `n_layers` in PatchTST backbone.
- `d_layers`: number of decoder layers, not used in PatchTST
- `d_ff`: Feed forward layer size
- `moving_avg`: Moving average window size, used in FEDformer, Autoformer etc
- `factor`: attention factor, not used in PatchTST
- `distil`: seems to only be used in Informer, "whether to use distilling in encoder, using this argument means not using distilling"
- `dropout`: dropout in other layers than head and fc
- `embed`: what kind of embedding used for -formers
- `activation`: what kind of activation function used. As the code is written right now I do not think this changes anything in PatchTST, as it uses `act` and not `activation`, which always take default value.

**Training**
- `num_workers`: the number of processes started when iterating over DataLoader (enumerate(dataloader)) Memory usage is number of workers * size of parent process.
- `itr`: How many times you run an experiment. (Probably for statistical significance)
- `train_epochs`: for how many epochs you want to train the network. One epoch means that the model has been trained on the all of the samples
-  `batch_size`: the number of samples that are in one training batch. The gradient is evaluated over the batch.
- `patience`: the number of epochs that are allowed to pass without a decrease in validation loss before early stopping.
- `learning_rate`: a coefficient for how much the parameters are updated from the gradient evaluated on each batch
- `des`: could be experiment description. 
- `loss`: what loss function to use. They seem to use MSE no matter what you put here
- `lradj`: what algo to use for updating learning rate. Options are: 'type{1:3}', 'constant', '3':'6', 'TST': 'TST' uses `torch.optim`:s `lr_scheduler` and the algo OneCycleLR. OneCycleLR starts at some lr, anneals it via some maximum lr to a minimum lr. The cycle is the total number of steps the training will take, in this case length of training data * number of epochs
- `pct_start`: (in OneCycleLR) the percentage of the full cycle that is spent increasing lr

**GPU**:
- `use_amp`: AMP, automatic mixed precision is a method that automatically casts values to suitable datatypes depending on use. Some operations require higher precision while other can be sped up. A cuda-method.
- `use_gpu`: enables GPU acceleration
- `gpu`: ?
- `use_multiple_gpu`: parallelises training on several GPUs.
- `devices`: a list of device ids of GPUs. if using multiple gpus this is needed   
- `test_flop`: gives number of trainable parameters and model complexity info.



