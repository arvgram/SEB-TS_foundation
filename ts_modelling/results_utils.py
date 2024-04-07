from PatchTST.PatchTST_supervised.utils.metrics import metric
import pandas as pd
import os


def write_to_metrics_csv(preds, trues, model_name, pretrain_data, train_head_data, finetune_data, test_data, folder_path):
    pt_data_name = ''
    for pt_file in pretrain_data:
        pt_filename = os.path.basename(pt_file)
        pt_data, suffix = os.path.splitext(pt_filename)
        pt_data_name += f'{pt_data}-{pretrain_data[pt_file]}__'
    pt_data_name = pt_data_name[:-2]

    ft_data_name = ''
    for ft_file in finetune_data:
        ft_filename = os.path.basename(ft_file)
        ft_data, suffix = os.path.splitext(ft_filename)
        ft_data_name += f'{ft_data}-{finetune_data[ft_file]}__'
    ft_data_name = ft_data_name[:-2]

    th_data_name = ''
    for th_file in train_head_data:
        th_filename = os.path.basename(th_file)
        th_data_, suffix = os.path.splitext(th_filename)
        th_data_name += f'{th_data_}-{train_head_data[th_file]}__'
    th_data_name = th_data_name[:-2]

    test_data_name = os.path.basename(test_data)
    test_data_name, suffix = os.path.splitext(test_data_name)

    mae, mse, rmse, mape, mspe, rse, _ = metric(preds, trues)  # corr is weird and broken
    metrics_df = pd.DataFrame({
        'model_name': [model_name],
        'pretrain_data': [pt_data_name],
        'train_head_data': [th_data_name],
        'finetune_data': [ft_data_name],
        'test_data': [test_data_name],
        'mae': [mae],
        'mse': [mse],
        'rmse': [rmse],
        'mape': [mape],
        'mspe': [mspe],
        'RSE': [rse],
    })

    # Check if CSV file already exists
    csv_path = os.path.join(folder_path, 'metrics.csv')
    if os.path.exists(csv_path):
        # Append DataFrame to existing CSV file
        metrics_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # Create new CSV file and write DataFrame
        metrics_df.to_csv(csv_path, index=False)
