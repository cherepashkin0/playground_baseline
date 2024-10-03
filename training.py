import pandas as pd
from gc import collect
import numpy as np
from sklearn.model_selection import PredefinedSplit as PDS
from tqdm import tqdm
from sklearn.base import clone
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import roc_auc_score
from myimports import PrintColor
from matplotlib import pyplot as plt
from colorama import Fore, Style, init
import ctypes
libc = ctypes.CDLL("libc.so.6")
from os import path, walk, getpid
import wandb



class Utils:
    """
    This class creates and uses several utility methods to be used across the code
    """;

    def __init__(self):
        pass

    def ScoreMetric(self, ytrue, ypred)-> float:
        """
        This method calculates the metric for the competition
        Inputs- ytrue, ypred:- input truth and predictions
        Output- float:- competition metric
        """;
        return roc_auc_score(ytrue, ypred)

    def CleanMemory(self):
        "This method cleans the memory off unused objects and displays the cleaned state RAM usage"

        collect();
        libc.malloc_trim(0)
        pid        = getpid()
        py         = Process(pid)
        memory_use = py.memory_info()[0] / 2. ** 30
        return f"\nRAM usage = {memory_use :.4} GB"

    def DisplayAdjTbl(self, *args):
        """
        This function displays pandas tables in an adjacent manner, sourced from the below link-
        https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
        """

        html_str = ''
        for df in args:
            html_str += df.to_html()
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)
        collect()

    def DisplayScores(
        self, Scores: pd.DataFrame, TrainScores: pd.DataFrame, methods: list
    ):
        "This method displays the scores and their means"

        args = \
        [Scores.style.format(precision = 5).\
         background_gradient(cmap = "Blues", subset = methods + ["Ensemble"]).\
         set_caption(f"\nOOF scores across methods and folds\n"),

         TrainScores.style.format(precision = 5).\
         background_gradient(cmap = "Pastel2", subset = methods).\
         set_caption(f"\nTrain scores across methods and folds\n")
        ];

        PrintColor(f"\n\n\n---> OOF score across all methods and folds\n",
                   color = Fore.LIGHTMAGENTA_EX
                   )
        self.DisplayAdjTbl(*args)

        print('\n')
        display(Scores.mean().to_frame().\
                transpose().\
                style.format(precision = 5).\
                background_gradient(cmap = "mako", axis=1,
                                    subset = Scores.columns
                                   ).\
                set_caption(f"\nOOF mean scores across methods and folds\n")
               )


utils = Utils()
collect()
print()

def MakePermImp(
        method, mdl, X, y, ygrp,
        myscorer,
        n_repeats = 2,
        state = 42,
        ntop: int = 15,
        **params,
):
    """
    This function makes the permutation importance for the provided model and returns the importance scores for all features

    Note-
    myscorer - scikit-learn -> metrics -> make_scorer object with the corresponding eval metric and relevant details
    """

    cv        = PDS(ygrp)
    n_splits  = ygrp.nunique()
    drop_cols = ["Source", "id", "Id", "Label", "fold_nb"]

    for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):
        Xtr  = X.iloc[train_idx].drop(drop_cols, axis=1, errors = "ignore")
        Xdev = X.iloc[dev_idx].drop(drop_cols, axis=1, errors = "ignore")
        ytr  = y.loc[Xtr.index]
        ydev = y.loc[Xdev.index]

        model = clone(mdl)
        sel_cols = list(Xdev.columns)
        model.fit(Xtr, ytr)

        imp_ = permutation_importance(model,
                                      Xdev, ydev,
                                      scoring = myscorer,
                                      n_repeats = n_repeats,
                                      random_state = state,
                                      )["importances_mean"]
        imp_ = pd.Series(index = sel_cols, data = imp_)

        display(
            imp_.\
            sort_values(ascending = False).\
            head(ntop).\
            to_frame().\
            transpose().\
            style.\
            format(formatter = '{:,.3f}').\
            background_gradient("icefire", axis=1).\
            set_caption(f"Top {ntop} features")
            )

        return imp_

class ModelTrainer:
    "This class trains the provided model on the train-test data and returns the predictions and fitted models"

    def __init__(
        self,
        problem_type   : str   = "binary",
        es             : int   = 100,
        target         : str   = "",
        metric_lbl     : str   = "auc",
        orig_req       : bool  = False,
        orig_all_folds : bool  = False,
        drop_cols      : list  = ["Source", "id", "Id", "Label", "fold_nb"],
    ):
        """
        Key parameters-
        es_iter - early stopping rounds for boosted trees
        """

        self.problem_type   = problem_type
        self.es_iter        = es
        self.target         = target
        self.drop_cols      = drop_cols + [self.target]
        self.metric_lbl     = metric_lbl
        self.orig_req       = orig_req
        self.orig_all_folds = orig_all_folds

    def ScoreMetric(self, ytrue, ypred):
        """
        This is the metric function for the competition scoring
        """
        return roc_auc_score(ytrue, ypred)

    def PlotFtreImp(
        self,
        ftreimp: pd.Series,
        method: str,
        ntop: int = 50,
        title_specs: dict = {'fontsize': 9,'fontweight' : 'bold','color': '#992600'},
        **params,
    ):
        "This function plots the feature importances for the model provided"

        print()
        fig, ax = plt.subplots(1, 1, figsize = (25, 7.5))

        ftreimp.sort_values(ascending = False).\
        head(ntop).\
        plot.bar(ax = ax, color = "blue")
        ax.set_title(f"Feature Importances - {method}", **title_specs)

        plt.tight_layout()
        # plt.show()
        fig.savefig(f"figs/Feature_Importances_{method}")
        print()

    def PostProcessPreds(self, ypred):
        "This method post-processes predictions optionally"
        return np.clip(ypred, a_min = 0.0, a_max = 1.0)

    def LoadData(
            self, X, y, Xtest,
            train_idx : list = [],
            dev_idx   : list = [],
            ):
        "This method loads the train and test data for the model fold using/ not using the original data"

        if self.orig_req == False:
            Xtr  = X.iloc[train_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

        elif self.orig_req == True and self.orig_all_folds == True:
            Xtr  = X.iloc[train_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

            orig_x = X.query("Source == 'Original'")[Xtr.columns]
            orig_y = y.iloc[orig_x.index]

            Xtr = pd.concat([Xtr, orig_x], axis = 0, ignore_index = True)
            ytr = pd.concat([ytr, orig_y], axis = 0, ignore_index = True)

        elif self.orig_req == True and self.orig_all_folds == False:
            Xtr  = X.iloc[train_idx].drop(self.drop_cols, axis=1, errors = "ignore")
            ytr  = y.iloc[Xtr.index]
            Xdev = X.iloc[dev_idx].query("Source == 'Competition'").drop(self.drop_cols, axis=1, errors = "ignore")
            ydev = y.iloc[Xdev.index]

        Xt = Xtest[Xdev.columns]

        # print(f"\n---> Shapes = {Xtr.shape} {ytr.shape} -- {Xdev.shape} {ydev.shape} -- {Xt.shape}")
        return (Xtr, ytr, Xdev, ydev, Xt)

    def MakePreds(self, X, fitted_model):
        "This method creates the model predictions based on the model provided, with optional post-processing"

        if self.problem_type == "regression":
            return self.PostProcessPreds(fitted_model.predict(X))
        elif self.problem_type == "binary":
            return self.PostProcessPreds(fitted_model.predict_proba(X)[:, 1])
        elif self.problem_type == "multiclass":
            return self.PostProcessPreds(fitted_model.predict_proba(X))

    def MakeOrigPreds(
            self, orig: pd.DataFrame, fitted_models: list, n_splits : int, ygrp: pd.Series,
            ):
        "This method creates the original data predictions separately only if required"

        if self.orig_req == False:
            orig_preds = 0

        elif self.orig_req == True and self.orig_all_folds == True:
            orig_preds = 0
            df = orig.drop(self.drop_cols, axis = 1, errors = "ignore")

            for fitted_model in fitted_models:
                orig_preds = orig_preds + (self.MakePreds(df, fitted_model) / n_splits)

        elif self.orig_req == True and self.orig_all_folds == False:
            len_orig   = orig.shape[0]
            orig.index = range(len_orig)
            orig_ygrp  = ygrp[-1 * len_orig:]
            orig_ygrp.index = range(len_orig)

            orig_preds = np.zeros(len_orig)
            for fold_nb, fitted_model in enumerate(fitted_models):
                df = \
                orig.iloc[orig_ygrp.loc[orig_ygrp == fold_nb].index].\
                drop(self.drop_cols, axis=1, errors = "ignore")

                orig_preds[df.index] = self.MakePreds(df, fitted_model)
                del df
        return orig_preds

    def MakeOfflineModel(
        self, X, y, ygrp, Xtest, mdl, method,
        test_preds_req   : bool = True,
        ftreimp_plot_req : bool = True,
        ntop             : int  = 50,
        **params,
    ):
        """
        This function trains the provided model on the dataset and cross-validates appropriately

        Inputs-
        X, y, ygrp       - training data components (Xtrain, ytrain, fold_nb)
        Xtest            - test data (optional)
        model            - model object for training
        method           - model method label
        test_preds_req   - boolean flag to extract test set predictions
        ftreimp_plot_req - boolean flag to plot tree feature importances
        ntop             - top n features for feature importances plot

        Returns-
        oof_preds, test_preds - prediction arrays
        fitted_models         - fitted model list for test set
        ftreimp               - feature importances across selected features
        mdl_best_iter         - model average best iteration across folds
        """

        oof_preds     = np.zeros(len(X.loc[X.Source == "Competition"]))
        orig_preds    = np.zeros(len(X.loc[X.Source == "Original"]))
        test_preds    = []
        mdl_best_iter = []
        ftreimp       = 0

        scores, tr_scores, fitted_models = [], [], []

        if self.orig_req == True:
            cv = PDS(ygrp)
        elif self.orig_req == False:
            X  = X.loc[X.Source == "Competition"]
            y  = y.iloc[X.index]
            cv = PDS(ygrp.iloc[0 : len(X)])

        n_splits = ygrp.nunique()

        for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y))):
            Xtr, ytr, Xdev, ydev, Xt = \
            self.LoadData(X, y, Xtest, train_idx, dev_idx)

            model = clone(mdl)

            if "CB" in method:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          verbose = 0,
                          early_stopping_rounds = self.es_iter,
                          )
                best_iter = model.get_best_iteration()

            elif "LGB" in method:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          callbacks = [log_evaluation(0),
                                       early_stopping(stopping_rounds = self.es_iter, verbose = False,),
                                       ],
                          )
                best_iter = model.best_iteration_

            elif "XGB" in method:
                model.fit(Xtr, ytr,
                          eval_set = [(Xdev, ydev)],
                          verbose  = 0,
                          )
                best_iter = model.best_iteration

            else:
                model.fit(Xtr, ytr)
                best_iter = -1

            fitted_models.append(model)

            try:
                ftreimp += model.feature_importances_
            except:
                pass

            dev_preds = self.MakePreds(Xdev, model)
            oof_preds[Xdev.index] = dev_preds

            train_preds  = self.MakePreds(Xtr, model)
            tr_score     = self.ScoreMetric(ytr.values.flatten(), train_preds)
            score        = self.ScoreMetric(ydev.values.flatten(), dev_preds)

            scores.append(score)
            tr_scores.append(tr_score)

            nspace = 15 - len(method) - 2 if fold_nb <= 9 else 15 - len(method) - 1
            PrintColor(f"{method} Fold{fold_nb} {' ' * nspace} OOF = {score:.6f} | Train = {tr_score:.6f} | Iter = {best_iter:,.0f} ")
            # wandb.log({f"{method}_OOF": score, f"{method}_Train": tr_score})
            mdl_best_iter.append(best_iter)

            if test_preds_req:
                test_preds.append(self.MakePreds(Xt, model))
            else:
                pass

        test_preds    = np.mean(np.stack(test_preds, axis = 1), axis=1)
        ftreimp       = pd.Series(ftreimp, index = Xdev.columns)
        mdl_best_iter = np.uint16(np.amax(mdl_best_iter))

        if ftreimp_plot_req :
            print()
            self.PlotFtreImp(ftreimp, method = method, ntop = ntop,)
        else:
            pass

        PrintColor(f"\n---> {np.mean(scores):.6f} +- {np.std(scores):.6f} | OOF", color = Fore.RED)
        PrintColor(f"---> {np.mean(tr_scores):.6f} +- {np.std(tr_scores):.6f} | Train", color = Fore.RED)
        scores_all = [np.mean(scores), np.mean(tr_scores)]
        # wandb.log({f"{method}_OOF": np.mean(score), f"{method}_Train": np.mean(tr_score)})
        if mdl_best_iter < 0:
            pass
        else:
            PrintColor(f"---> Max best iteration = {mdl_best_iter :,.0f}", color = Fore.RED)

        if self.orig_req:
            print(f"---> Collecting original predictions")
            orig_preds = self.MakeOrigPreds(X.loc[X.Source == "Original"],
                                            fitted_models,
                                            n_splits,
                                            ygrp,
                                            )
            oof_preds = np.concatenate([oof_preds, orig_preds], axis= 0)
        else:
            pass

        return (fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter, scores_all)

    def MakeOnlineModel(
        self, X, y, Xtest, model, method,
        test_preds_req : bool = False,
    ):
        "This method refits the model on the complete train data and returns the model fitted object and predictions"

        try:
            model.early_stopping_rounds = None
        except:
            pass

        try:
            model.fit(X, y, verbose = 0)
        except:
            model.fit(X, y,)

        oof_preds  = model.predict(X)
        if test_preds_req:
            test_preds = model.predict(Xtest[X.columns])
        else:
            test_preds = 0
        return (model, oof_preds, test_preds)

class OptunaEnsembler:
    """
    This is the Optuna ensemble class-
    Source- https://www.kaggle.com/code/arunklenin/ps3e26-cirrhosis-survial-prediction-multiclass
    """;

    def __init__(
        self, state: int = 42, ntrials: int = 300, metric_obj: str = "minimize",
        **params
    ):
        self.study        = None
        self.weights      = None
        self.random_state = state
        self.n_trials     = ntrials
        self.direction    = metric_obj

    def ScoreMetric(self, ytrue, ypred):
        """
        This is the metric function for the competition
        """;
        return roc_auc_score(ytrue, ypred)

    def _objective(
        self, trial, y_true, y_preds
    ):
        """
        This method defines the objective function for the ensemble
        """;

        if isinstance(y_preds, pd.DataFrame) or isinstance(y_preds, np.ndarray):
            weights = [trial.suggest_float(f"weight{n}", 0.001, 0.999)
                       for n in range(y_preds.shape[-1])
                      ]
            axis = 1

        elif isinstance(y_preds, list):
            weights = [trial.suggest_float(f"weight{n}", 0.001, 0.999)
                       for n in range(len(y_preds))
                      ]
            axis = 0

        # Calculating the weighted prediction:-
        weighted_pred  = np.average(np.array(y_preds), axis = axis, weights = weights)
        score          = self.ScoreMetric(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        "This method fits the Optuna objective on the fold level data";

        optuna.logging.set_verbosity = optuna.logging.ERROR

        self.study = \
        optuna.create_study(sampler    = TPESampler(seed = self.random_state),
                            pruner     = HyperbandPruner(),
                            study_name = "Ensemble",
                            direction  = self.direction,
                           )

        obj = partial(self._objective, y_true = y_true, y_preds = y_preds)
        self.study.optimize(obj, n_trials = self.n_trials)

        if isinstance(y_preds, list):
            self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

        else:
            self.weights = [self.study.best_params[f"weight{n}"] for n in range(y_preds.shape[-1])]

    def predict(self, y_preds):
        "This method predicts using the fitted Optuna objective";

        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict';

        if isinstance(y_preds, list):
            weighted_pred = np.average(np.array(y_preds), axis=0, weights = self.weights)

        else:
            weighted_pred = np.average(np.array(y_preds), axis=1, weights = self.weights)

        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        """
        This method fits the Optuna objective on the fold data, then predicts the test set
        """;
        self.fit(y_true, y_preds)
        return self.predict(y_preds)

    def weights(self):
        "This method returns the non-normalized weights for all models in a fold"
        return self.weights

print()
collect();

def NormWeights(weights: dict, methods: list):
    "This function normalizes the weights and returns a dataframe of normalized weights across folds and models"

    weights = pd.DataFrame.from_dict(weights).T
    weights["row_sum"] = weights.sum(axis=1)

    for col in weights.columns:
        weights[col] = weights[col] / weights["row_sum"]

    weights.drop("row_sum", axis = 1, inplace = True, errors = "ignore")
    weights.columns    = methods
    weights.index.name = "Fold_Nb"
    return weights

def MakeEnsemble(target: str, ntrials: int = 300):
    "This function implements the Optuna ensemble on the OOF and test prediction datasets"

    global OOF_Preds, Mdl_Preds

    PrintColor(f"\n{'=' * 20} ENSEMBLE {'=' * 20}\n")

    ygrp       = OOF_Preds["fold_nb"]
    cv         = PDS(ygrp)
    oof_preds  = np.zeros(len(OOF_Preds))
    test_preds = []
    scores     = []
    weights    = {}
    drop_cols  = ["fold_nb", target, "Ensemble"]
    n_splits   = ygrp.nunique()

    for fold_nb, (_, dev_idx) in tqdm(enumerate(cv.split(OOF_Preds, OOF_Preds[target]))):
        Xdev = OOF_Preds.iloc[dev_idx].drop(drop_cols, axis=1, errors = "ignore")
        ydev = OOF_Preds.loc[dev_idx, target]

        ens = OptunaEnsembler(ntrials = ntrials)
        ens.fit(ydev, Xdev,)

        dev_preds = ens.predict(Xdev)
        score     = ens.ScoreMetric(ydev.values, dev_preds)
        oof_preds[dev_idx] = dev_preds
        test_preds.append(
            ens.predict(Mdl_Preds.drop(drop_cols, axis=1, errors = "ignore"))
        )

        PrintColor(f"---> {score: .6f} | Fold {fold_nb}", color = Fore.CYAN)
        scores.append(score)

        weights[f"Fold{fold_nb}"] = ens.weights

    PrintColor(f"\n---> OOF = {np.mean(scores): .6f} +- {np.std(scores): .6f} | Ensemble",
               color = Fore.RED
              )

    test_preds = np.mean(np.stack(test_preds, axis=1), axis=1,)

    OOF_Preds["Ensemble"] = oof_preds
    Mdl_Preds["Ensemble"] = test_preds

    weights = \
    NormWeights(
        weights,
        methods = Mdl_Preds.drop(drop_cols, axis=1, errors = "ignore").columns
    )

    print("\n\n\n")
    display(
        weights.\
        style.\
        set_caption("Normalized weights").\
        format(precision = 6).\
        set_properties(
            props = "color:red; background-color:white; font-weight: bold; border: maroon dashed 1.6px"
        )
    )

    return weights
