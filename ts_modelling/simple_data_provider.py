import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
    A simple dataset for use with PyTorch and training time series models. Does not use datetime/frequency features.
    """

    def __init__(self, root_path, data_path, flag, features, target, seq_len, pred_len, train_share=0.6,
                 test_share=0.2):
        self.path = root_path + '/' + data_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.train_share = train_share
        self.test_share = test_share

        assert features in ['M', 'S', 'MS']

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.path)
        if self.features == 'S':
            cols_of_interest = [self.target]
        else:
            cols = df_raw.columns.tolist()
            if 'date' in cols:
                cols.remove('date')
            if self.features == 'MS':
                cols.remove(self.target)
                cols_of_interest = [cols + [self.target]]
            else:
                cols_of_interest = cols

        num_train = int(len(df_raw) * self.train_share)
        num_test = int(len(df_raw) * self.test_share)
        num_val = len(df_raw) - num_test - num_train

        left_border_idx = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        right_border_idx = [num_train, num_train + num_val, len(df_raw)]
        left_border = left_border_idx[self.set_type]
        right_border = right_border_idx[self.set_type]

        self.cols = cols_of_interest
        self.data_x = df_raw.loc[left_border:right_border, cols_of_interest].values
        self.data_y = df_raw.loc[left_border:right_border, cols_of_interest].values

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len

        seq_x = self.data_x[x_begin:x_end]
        seq_y = self.data_y[y_begin:y_end]

        return seq_x, seq_y,


class MultiDataset(Dataset):
    """
    A dataset for sampling from multiple datasets.
    Parameters:
    - root_path: str path to the directory containing datsets
    - data_paths_target: dict {file_name: target_in_data} ex: {'temperature.csv': 'sthlm_temp'}
    - flag (str): A string indicating the dataset flag (e.g., 'train', 'val', 'test').
    - features (str): A string indicating the type of features ('M', 'S', 'MS').
    - seq_len (int): The input length.
    - pred_len (int): The prediction length.
    - train_share (float): The proportion of data to use for training.
    - test_share (float): The proportion of data to use for testing.
    """

    def __init__(self, root_path, data_paths_targets, flag, features, seq_len, pred_len, train_share=0.6,
                 test_share=0.2):
        self.data_paths_targets = data_paths_targets
        self.flag = flag
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_share = train_share
        self.test_share = test_share

        assert features in ['M', 'S', 'MS']

        self.datasets = []

        # Initialize SimpleDataset instances for each dataset
        for data_path, target in data_paths_targets.items():
            dataset = SimpleDataset(
                root_path=root_path,
                data_path=data_path,
                flag=flag,
                features=features,
                target=target,
                seq_len=seq_len,
                pred_len=pred_len,
                train_share=train_share,
                test_share=test_share
            )
            self.datasets.append(dataset)
        self.cols = [col for d in self.datasets for col in d.cols]

    def __len__(self):
        # length is length of all dataset, using __len__ in SimpleDataset class
        self.total_length = sum(len(dataset) for dataset in self.datasets)
        return self.total_length

    def __getitem__(self, index):
        # index is a "global" index and we thus need to find which dataset it corresponds to
        dataset_index = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_index += 1
            cumulative_length += len(self.datasets[dataset_index])

        # we subtract the length of the datasets we have iterated past to get local index
        local_index = index - (cumulative_length - len(self.datasets[dataset_index]))

        # sample from the chosen dataset
        return self.datasets[dataset_index][local_index]


class SimpleDataProvider:
    def __init__(self, args, flag):
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        self.dataset = MultiDataset(
            root_path=args.root_path,
            data_paths_targets=args.data_paths_targets,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            features=args.features,
        )
        print(flag, len(self.dataset))
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

    def get_cols(self):
        return self.dataset.cols

    def get_dataset_data_loader(self):
        return self.dataset, self.data_loader
