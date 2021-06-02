import csv
from pathlib import Path
import pandas as pd
import numpy as np
import torch as torch
from torch import nn
from torch.nn import functional as F
import sklearn
import time
import wandb
import uuid
import numerapi
import pathlib
import os
import shutil
from functools import reduce
import scipy
from fast_soft_sort.pytorch_ops import soft_rank
from tqdm import tqdm

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

def getImportantFeatures(model,test_data,feature_names):
    diff = MDA(model,feature_names,test_data)
    keep_features=[]
    for i in diff:
        if i[1] > 0:
            keep_features.append(i[0])
    return keep_features

def corrcoef(target, pred):
    # np.corrcoef in torch from @mdo
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

def spearman(
    target,
    pred,
    regularization="l2",
    regularization_strength=1.0,
):
    target = target.view(1,-1)
    pred = pred.view(1,-1)
    pred = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    return pred.requires_grad_(True).mean()

def score_(df):
    return correlation(df['prediction'], df['target'])  

def numerai_score(df):
    scores = df.groupby('era').apply(score_)
    return scores.mean(), scores.std(ddof=0)

def MDA(model, features, testSet):     
    """
    function from https://forum.numer.ai/t/feature-selection-by-marcos-lopez-de-prado/3170
    """
    preds=model(torch.from_numpy(testSet[features].to_numpy()).float().view(-1, 1, len(features))).detach().numpy()
    testSet['prediction'] = preds  # predict with a pre-fitted model on an OOS validation set
    corr, std = numerai_score(testSet)  # save base scores
    print("Base corr: ", corr)
    diff = []
    np.random.seed(42)
    with tqdm(total=len(features)) as progress:
        for col in features:   # iterate through each features
            X = testSet.copy()
            np.random.shuffle(X[col].values)    # shuffle the a selected feature column, while maintaining the distribution of the feature
            inp = torch.from_numpy(X[features].to_numpy()).view(-1, 1, len(features)).float()
            testSet['prediction'] = model(inp).detach().numpy()# run prediction with the same pre-fitted model, with one shuffled feature
            corrX, stdX = numerai_score(testSet)  # compare scores...
            # print(col, corrX-corr)
            diff.append((col, corrX-corr))
            progress.update(1)
    return diff

def refresh_numerai_data():
    remove("numerai_datasets.zip")
    remove("numerai_datasets")
    napi = numerapi.NumerAPI(verbosity="info")
    napi.download_current_dataset(unzip=True,dest_filename="numerai_datasets")

def get_factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def determine_fitness_for_batch_size(n):
    if n < 4000 and n > 80:
        return True
    else:
        return False

def get_batch_size(total_size):
    factors = list(get_factors(total_size))
    factors = list(filter(determine_fitness_for_batch_size, factors))
    if len(factors) > 0:
        return np.max(factors)
    else:
        return 1

def calculate_multilayer_output_length_conv(layers, length_in, kernel_size, stride=1, padding=0, dilation=1):
    for i in range(layers):
        length_in = calculate_output_length_conv(length_in, kernel_size, stride, padding, dilation)
    return length_in

def get_dataset():
    training_data = pd.read_csv("numerai_datasets/numerai_training_data.csv")
    target = training_data["target"]
    tournament_data = pd.read_csv("numerai_datasets/numerai_tournament_data.csv")
    feature_names = [
        f for f in training_data.columns if f.startswith("feature")
    ]
    return training_data,tournament_data,feature_names

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

class BatchNormResizeLayer(nn.Module):
    def __init__(self, lambd):
        super(BatchNormResizeLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class CustomConfig:
    """A class for storing configurations that works with wandb."""
    def __init__(self, init_dict=None, **kwargs):
        self.dict_version={}
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.dict_version[k]=v
        if init_dict != None:
            for i in init_dict:
                self.dict_version[i]=init_dict[i]
                setattr(self, i, init_dict[i])

def calculate_output_length_conv(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]

# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])

# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)
    
# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(df,
            columns,
            extra_neutralizers=None,
            proportion=1.0,
            normalize=True,
            era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)

# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
        np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized

def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)

def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                        feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: correlation(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)

def sample_val_corr(model,feature_names,tournament_data,sample_len=10000,features=310,filter_for_validation=True,shuffled=False):
    """gets the analytics and results for a snapshot of the train and tournament data. SHould take less than a minute to run."""
    if shuffled:
        tournament_data_test = tournament_data.sample(sample_len)
    else:
        tournament_data_test = tournament_data[:sample_len]
    with torch.no_grad():
        pred_tournament = torch.from_numpy(tournament_data_test[feature_names].to_numpy()).float()
        tourn_pred = []
        for batch in pred_tournament.view(-1, 100, 1, features):
            tourn_pred.append(model(batch).numpy())
        tourn_pred = np.array(tourn_pred).reshape(-1)
        tournament_data_test[PREDICTION_NAME] = tourn_pred
    if filter_for_validation:
    # Check the per-era correlations on the validation set (out of sample)
        validation_data = tournament_data_test[tournament_data_test.data_type == "validation"]
    else: 
        validation_data = tournament_data_test
    validation_correlations = validation_data.groupby("era").apply(score)
    return validation_correlations.mean()
    
def full_tourn_results(model,feature_names,tournament_data,split=1720000,pth="submission.csv"):
    """
    gets the tournament results and analytics using tournament data and predicts and outputs a submission file ready to be submitted
    Note: should take about 15 - 60 minutes to run depending on complexity of model
    """
    tournament_data_test = tournament_data.copy()
    print("Generating predictions...")
    with torch.no_grad():
        pred1 = torch.from_numpy(tournament_data_test[:1720000][feature_names].to_numpy()).float().view(-1, 5000, 1, len(feature_names))
        pred2 = torch.from_numpy(tournament_data_test[1720000:][feature_names].to_numpy()).float().view(-1, 1, 1, len(feature_names))
        tourn_pred = []
        with tqdm(total=len(pred1), leave=True, position=0) as progress:
            for batch in pred1:
                res=model(batch).detach().numpy().reshape(-1)
                for i in res:
                    tourn_pred.append(i)
                progress.update(1)
        with tqdm(total=len(pred2), leave=True, position=0) as progress:
            for batch in pred2:
                res=model(batch).detach().numpy().reshape(-1)
                for i in res: 
                    tourn_pred.append(i)
                progress.update(1)
        tourn_pred = np.array(tourn_pred).reshape(-1)
        tournament_data_test[PREDICTION_NAME] = tourn_pred

    # Check the per-era correlations on the validation set (out of sample)
    validation_data = tournament_data_test[tournament_data_test.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and "
            f"std {validation_correlations.std(ddof=0)}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

    # Check the "sharpe" ratio on the validation set
    validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
    print(f"Validation Sharpe: {validation_sharpe}")

    print("checking max drawdown...")
    rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                                    min_periods=1).max()
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    print(f"max drawdown: {max_drawdown}")

    # Check the feature exposure of your validation predictions
    feature_exposures = validation_data[feature_names].apply(lambda d: correlation(validation_data[PREDICTION_NAME], d),
                                                                axis=0)
    max_per_era = validation_data.groupby("era").apply(
        lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
    max_feature_exposure = max_per_era.mean()
    print(f"Max Feature Exposure: {max_feature_exposure}")

    # Check feature neutral mean
    print("Calculating feature neutral mean...")
    feature_neutral_mean = get_feature_neutral_mean(validation_data)
    print(f"Feature Neutral Mean is {feature_neutral_mean}")
    tournament_data_test=tournament_data_test.set_index('id')
    #update notion
    model.notion_model_page.val_corr = validation_correlations.mean()
    model.notion_model_page.val_sharp = validation_sharpe

    # Save predictions as a CSV and upload to https://numer.ai
    tournament_data_test[PREDICTION_NAME].to_csv(pth, header=True)