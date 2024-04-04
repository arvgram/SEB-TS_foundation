from PatchTST.PatchTST_supervised.utils.metrics import metric
import pandas as pd
import os


def write_to_metrics_csv(preds, trues, model_name, dataset, folder_path):
    filename = os.path.basename(dataset)
    data_name, suffix = os.path.splitext(filename)

    mae, mse, rmse, mape, mspe, rse, _ = metric(preds, trues)  # corr is weird and broken
    metrics_df = pd.DataFrame({
        'model_name': [model_name],
        'dataset': [data_name],
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
