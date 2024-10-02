
print(f"\n---> Commencing imports-part1")

from gc import collect
from warnings import filterwarnings
filterwarnings('ignore')
from IPython.display import display_html, clear_output
clear_output()
import os, sys, logging, re, joblib, ctypes, shutil
from copy import deepcopy

import xgboost as xgb, lightgbm as lgb, catboost as cb, sklearn as sk, pandas as pd
print(f"---> XGBoost = {xgb.__version__} | LightGBM = {lgb.__version__} | Catboost = {cb.__version__}")
print(f"---> Sklearn = {sk.__version__}| Pandas = {pd.__version__}")
collect()

# General library imports:-
from warnings import filterwarnings
filterwarnings('ignore')
from gc import collect

from os import path, walk, getpid
from psutil import Process
import re
from collections import Counter
from itertools import product

import ctypes
libc = ctypes.CDLL("libc.so.6")

from IPython.display import display_html, clear_output
from pprint import pprint
from functools import partial
from copy import deepcopy
import pandas as pd, numpy as np, os, joblib
import polars as pl
import polars.selectors as cs
import re

from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from colorama import Fore, Style, init
from warnings import filterwarnings
filterwarnings('ignore')
from tqdm.notebook import tqdm

print(f"---> Imports- part 1 done\n")

# Importing model and pipeline specifics:-
from category_encoders import OrdinalEncoder, OneHotEncoder

# Pipeline specifics:-
from sklearn.preprocessing import (RobustScaler,
                                   MinMaxScaler,
                                   StandardScaler,
                                   FunctionTransformer as FT,
                                   PowerTransformer,
                                  )
from sklearn.impute import SimpleImputer as SI
from sklearn.model_selection import (RepeatedStratifiedKFold as RSKF,
                                     StratifiedKFold as SKF,
                                     StratifiedGroupKFold as SGKF,
                                     KFold,
                                     GroupKFold as GKF,
                                     RepeatedKFold as RKF,
                                     PredefinedSplit as PDS,
                                     cross_val_score,
                                     cross_val_predict,
                                    )
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold as VT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer

# ML Model training:-
from sklearn.metrics import (roc_auc_score,
                             r2_score,
                             make_scorer,
                             brier_score_loss,
                             )

import xgboost as xgb, lightgbm as lgb
from xgboost import QuantileDMatrix, XGBClassifier as XGBC
from lightgbm import log_evaluation, early_stopping, LGBMClassifier as LGBMC
from catboost import CatBoostClassifier as CBC, Pool
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LRC

# Ensemble and tuning:-
import optuna
from optuna import Trial, trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

# Setting rc parameters in seaborn for plots and graphs-
sns.set({"axes.facecolor"       : "#ffffff",
         "figure.facecolor"     : "#ffffff",
         "axes.edgecolor"       : "#000000",
         "grid.color"           : "#ffffff",
         "font.family"          : ['Cambria'],
         "axes.labelcolor"      : "#000000",
         "xtick.color"          : "#000000",
         "ytick.color"          : "#000000",
         "grid.linewidth"       : 0.75,
         "grid.linestyle"       : "--",
         "axes.titlecolor"      : '#0099e6',
         'axes.titlesize'       : 8.5,
         'axes.labelweight'     : "bold",
         'legend.fontsize'      : 7.0,
         'legend.title_fontsize': 7.0,
         'font.size'            : 7.5,
         'xtick.labelsize'      : 7.5,
         'ytick.labelsize'      : 7.5,
        }
       )

# Color printing
def PrintColor(text: str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string"
    print(style + color + text + Style.RESET_ALL)

print(f"---> Commencing imports-part2")
optuna.logging.set_verbosity = optuna.logging.ERROR
optuna.logging.disable_default_handler()
print(f"---> XGBoost = {xgb.__version__} | LightGBM = {lgb.__version__}")

##################################################################
# Customizing logging for LGBM
class MyLogger:
    """
    This class helps to suppress logs in lightgbm and Optuna
    Source - https://github.com/microsoft/LightGBM/issues/6014
    """

    def init(self, logging_lbl: str):
        self.logger = logging.getLogger(logging_lbl)
        self.logger.setLevel(logging.ERROR)

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        self.logger.error(message)

l = MyLogger()
l.init(logging_lbl = "lightgbm_custom")
lgb.register_logger(l)

##################################################################
# Customizing logging for XGBoost
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(f'xgb_optimize.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

class XGBLogging(xgb.callback.TrainingCallback):
    """log train logs to file"""

    def __init__(self, epoch_log_interval=100):
        self.epoch_log_interval = epoch_log_interval

    def after_iteration(self, model, epoch:int,
                        evals_log:xgb.callback.TrainingCallback.EvalsLog
                        ):

        if self.epoch_log_interval <= 0:
            pass

        elif (epoch %  self.epoch_log_interval == 0):
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                    logger.info(f"XGBLogging epoch {epoch} dataset {data} {metric_name} {score}")

        return False

# Making sklearn pipeline outputs as dataframe:-
from sklearn import set_config
# set_config(transform_output = "polars")
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 200)
print(f"---> Imports- part 2 done")

print(f"\n---> Imports done")
