from ts_modelling.utils.metrics_utils import metrics
import pandas as pd
import os


def write_to_metrics_csv(preds, trues, model_type, model_name, train_history, test_data, target, folder_path):
    history = ''
    for entry in train_history:
        h = '_'.join([os.path.splitext(data)[0] + '-' + targets for (data, targets) in entry['data_targets'].items()])
        h = h + '.' + str(entry['epochs'])
        h = h + '.' + entry['training_task'] + '__'
        history = history + h
    history = history[:-2]

    test_data_name = os.path.basename(test_data)
    test_data_name, suffix = os.path.splitext(test_data_name)

    mae, mse, rmse, _, mspe, _, nrv = metrics(preds, trues)
    metrics_df = pd.DataFrame({
        'model_name': [model_name],
        'model_type': [model_type],
        'train_history': [history],
        'test_data': [test_data_name],
        'target': [target],
        'mae': [mae],
        'mse': [mse],
        'rmse': [rmse],
        'mspe': [mspe],
        'nrv': [nrv],
    })

    # Check if CSV file already exists
    csv_path = os.path.join(folder_path, 'metrics.csv')
    if os.path.exists(csv_path):
        # Append DataFrame to existing CSV file
        metrics_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # Create new CSV file and write DataFrame
        metrics_df.to_csv(csv_path, index=False)


def plot_preds(test_results_path, nbr_plots=3, show=True):
    import numpy as np
    from matplotlib import pyplot as plt
    import yaml

    # output path is where the test predictions are located
    output_path = os.path.join(test_results_path, 'input_pred_true')
    plot_path = os.path.join(test_results_path, 'plots/')
    os.makedirs(plot_path, exist_ok=True)

    for folder_name in os.listdir(output_path):
        current_dir = os.path.join(output_path, folder_name)
        preds = np.load(current_dir + '/pred.npy')
        trues = np.load(current_dir + '/true.npy')
        inputs = np.load(current_dir + '/input.npy')

        with open(current_dir + '/data_names.yaml', 'r') as file:
            data_names = yaml.safe_load(file)

        current_plot_path = os.path.join(plot_path, folder_name)
        os.makedirs(current_plot_path, exist_ok=True)

        interval = trues.shape[0] // nbr_plots
        for i in range(nbr_plots):
            for j, col in enumerate(data_names['columns']):
                idx = i * interval

                y = trues[idx, :, j]
                yhat = preds[idx, :, j]
                x = inputs[idx, :, j]

                plt.figure()

                plt.plot(x, label='input')
                plt.plot(range(len(x), len(x) + len(y)), y, label='true', alpha=0.5)
                plt.plot(range(len(x), len(x) + len(y)), yhat, label='pred')
                plt.title(f'Predictions for variable: {col}, data: {folder_name}')
                plt.legend()
                plt.savefig(
                    os.path.join(current_plot_path, f'data-{folder_name}_channel-{col}_batch-{i}.pdf'),
                    format='pdf')
                if show:
                    plt.show()
                plt.close()
