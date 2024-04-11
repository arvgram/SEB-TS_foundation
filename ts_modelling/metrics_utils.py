import numpy as np


def _rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def _mae(pred, true):
    return np.mean(np.abs(pred - true))


def _mse(pred, true):
    return np.mean((pred - true) ** 2)


def _rmse(pred, true):
    return np.sqrt(_mse(pred, true))


def _mape(pred, true):
    return np.mean(np.abs((pred - true) / true))


def _mspe(pred, true):
    return np.mean(np.square((pred - true) / true))


def norm_res_var(preds, trues):
    """the residual variance normalised with true values variance"""
    # var(y-yhat)/var(y)
    return np.var(preds-trues)/np.var(trues)


def metrics(pred, true):
    mae = _mae(pred, true)
    mse = _mse(pred, true)
    rmse = _rmse(pred, true)
    mape = _mape(pred, true)
    mspe = _mspe(pred, true)
    rse = _rse(pred, true)
    nrv = norm_res_var(pred, true)

    return mae, mse, rmse, mape, mspe, rse, nrv
