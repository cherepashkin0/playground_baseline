import pandas as pd
from myimports import *
from training import *
from mypp import *
from gc import collect
from config import *
from pathlib import Path

cv_selector = \
{
 "RKF"   : RKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "RSKF"  : RSKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
 "SKF"   : SKF(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
 "GKF"   : GKF(n_splits = CFG.n_splits)
}
# collect()
pp = Preprocessor()
pp.DoPreprocessing()
cat_mdl_cols = \
    list(pp.test.select_dtypes([np.int8, np.int16, np.int32, np.int64]).columns) + pp.cat_cols

def data_transform():

    ytrain = pp.train[CFG.target]
    Xtrain = pp.train.drop([CFG.target, CFG.grouper], axis= 1, errors = "ignore")
    PrintColor(f"---> Shapes = {Xtrain.shape} {ytrain.shape} {pp.test.shape}")

    # cat_mdl_cols = \
    # list(pp.test.select_dtypes([np.int8, np.int16, np.int32, np.int64]).columns) + pp.cat_cols

    Xtrain = MakeFtre(Xtrain, cat_mdl_cols)
    Xtest  = MakeFtre(pp.test, cat_mdl_cols)

    PrintColor(f"---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape}")

    # Initializing the cv scheme:-
    cv = cv_selector[CFG.mdlcv_mthd]

    if CFG.nb_orig > 0:
        all_df = []

        for mysource in ["Competition", "Original"]:
            df = pd.concat([Xtrain.loc[Xtrain.Source == mysource], ytrain], axis=1, join = "inner")
            df.index = range(len(df))
            for fold_nb, (_, dev_idx) in enumerate(cv.split(df, df[CFG.target])):
                df.loc[dev_idx, "fold_nb"] = fold_nb

            all_df.append(df)
        ygrp = pd.concat(all_df, axis=0, ignore_index = True)["fold_nb"].astype(np.uint8)

    else:
        df = Xtrain.loc[Xtrain.Source == "Competition"]
        df.index = range(len(df))

        for fold_nb, (_, dev_idx) in enumerate(cv.split(df, ytrain.iloc[df.index])):
            df.loc[dev_idx, "fold_nb"] = fold_nb
        ygrp = df["fold_nb"].astype(np.uint8)
    return Xtrain, Xtest, ytrain, ygrp

def MakeFtre(X: pd.DataFrame, cat_cols: list):
    "This function makes extra features for the dataset provided"

    df = X.copy()
    df["loantoincome"] = (df["loan_amnt"] / df["person_income"]) - df["loan_percent_income"]

    df[cat_cols] = df[cat_cols].astype("category")
    return df


def model_training():
    try:
        l = MyLogger()
        l.init(logging_lbl = "lightgbm_custom")
        lgb.register_logger(l)
    except:
        pass

    # Initializing model parameters
    Mdl_Master = \
    {
     f'LGBM1C' : LGBMC(**{"objective"           : "binary",
                          "metrics"             : "auc",
                          'device'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                          'learning_rate'       : 0.0325,
                          'n_estimators'        : 5_000,
                          'max_depth'           : 7,
                          'num_leaves'          : 25,
                          'min_data_in_leaf'    : 20,
                          'feature_fraction'    : 0.70,
                          'bagging_fraction'    : 0.88,
                          'bagging_freq'        : 6,
                          'lambda_l1'           : 0.001,
                          'lambda_l2'           : 0.1,
                          'verbosity'           : -1,
                          'random_state'        : CFG.state,
                         }
                      ),

     f'LGBM2C' : LGBMC(**{"objective"           : "binary",
                          "metrics"             : "auc",
                          'device'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",
                          'learning_rate'       : 0.035,
                          'data_sample_strategy': 'goss',
                          'n_estimators'        : 5_000,
                          'max_depth'           : 7,
                          'num_leaves'          : 30,
                          'min_data_in_leaf'    : 30,
                          'feature_fraction'    : 0.60,
                          'colsample_bytree'    : 0.65,
                          'lambda_l1'           : 0.001,
                          'lambda_l2'           : 1.25,
                          'verbosity'           : -1,
                          'random_state'        : CFG.state,
                         }
                      ),


     f'XGB1C' : XGBC(**{  "objective"             : "binary:logistic",
                          "eval_metric"           : "auc",
                          'device'                : "cuda" if CFG.gpu_switch == "ON" else "cpu",
                          'learning_rate'         : 0.03,
                          'n_estimators'          : 5_000,
                          'max_depth'             : 7,
                          'colsample_bytree'      : 0.75,
                          'colsample_bynode'      : 0.85,
                          'colsample_bylevel'     : 0.45,
                          'reg_alpha'             : 0.001,
                          'reg_lambda'            : 0.25,
                          'verbose'               : 0,
                          'random_state'          : CFG.state,
                          'enable_categorical'    : True,
                          'callbacks'             : [XGBLogging(epoch_log_interval= 0)],
                          'early_stopping_rounds' : CFG.nbrnd_erly_stp,
                         }
                      ),

     f'CB1C' : CBC(**{'task_type'           : "CPU",
                      'loss_function'       : 'Logloss',
                      'eval_metric'         : "AUC",
                      'bagging_temperature' : 0.25,
                      'colsample_bylevel'   : 0.40,
                      'iterations'          : 5_000,
                      'learning_rate'       : 0.045,
                      'max_depth'           : 7,
                      'l2_leaf_reg'         : 0.80,
                      'min_data_in_leaf'    : 30,
                      'random_strength'     : 0.25,
                      'verbose'             : 0,
                      'cat_features'        : cat_mdl_cols,
                     }
                  ),
    }

    return Mdl_Master

def single_model(Mdl_Master, Xtrain, Xtest, ytrain, ygrp):
    # Initializing model outputs
    OOF_Preds    = {}
    Mdl_Preds    = {}
    FittedModels = {}
    FtreImp      = {}
    SelMdlCols   = {}
    # %%time

    # Model training:-
    drop_cols = ["Source", "id", "Id", "Label", CFG.target, "fold_nb"]

    for method, mymodel in tqdm(Mdl_Master.items()):

        PrintColor(f"\n{'=' * 20} {method.upper()} MODEL TRAINING {'=' * 20}\n")

        md = \
        ModelTrainer(
            problem_type   = "binary",
            es             = CFG.nbrnd_erly_stp,
            target         = CFG.target,
            orig_req       = True if CFG.nb_orig > 0 else False,
            orig_all_folds = CFG.orig_all_folds,
            metric_lbl     = "auc",
            drop_cols      = drop_cols,
            )

        sel_mdl_cols = list(Xtest.columns)

        PrintColor(f"Selected columns = {len(sel_mdl_cols) :,.0f}", color = Fore.RED)
        SelMdlCols[method] = sel_mdl_cols, cat_mdl_cols

        fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter =  \
        md.MakeOfflineModel(
            Xtrain[sel_mdl_cols].copy(),
            deepcopy(ytrain),
            ygrp,
            Xtest[sel_mdl_cols].copy(),
            clone(mymodel),
            method,
            test_preds_req   = True,
            ftreimp_plot_req = CFG.ftre_imp_req,
            ntop = 50,
            # ntop = 50,
        )

        OOF_Preds[method]    = oof_preds
        Mdl_Preds[method]    = test_preds
        FittedModels[method] = fitted_models
        FtreImp[method]      = ftreimp

        del fitted_models, oof_preds, test_preds, ftreimp, sel_mdl_cols
        print()
        collect();

    _ = utils.CleanMemory();

def ensemble():
    # %%time

    len_train  = len(Xtrain.loc[Xtrain.Source == "Competition"])
    oof_preds  = pd.DataFrame.from_dict(OOF_Preds, orient = "columns").iloc[0 : len_train]
    oof_preds[CFG.target] = ytrain.iloc[0 : len_train]
    oof_preds["fold_nb"]  = ygrp.iloc[0 : len_train]

    mdl_preds  = pd.DataFrame.from_dict(Mdl_Preds, orient = "columns")
    sel_cols   = mdl_preds.columns

    test_preds = 0
    scores     = 0
    ens_preds  = np.zeros(len(oof_preds))
    mycv       = PDS(ygrp.iloc[0 : len_train])

    PrintColor(f"\n ----- Logistic Ensemble ----- \n")

    for fold_nb, (train_idx, dev_idx) in enumerate(mycv.split(oof_preds, oof_preds[CFG.target])):
        Xtr  = oof_preds.iloc[train_idx][sel_cols]
        ytr  = oof_preds.loc[train_idx, CFG.target]
        Xdev = oof_preds.iloc[dev_idx][sel_cols]
        ydev = oof_preds.loc[dev_idx, CFG.target]

        model = LRC(C = 0.10, random_state = CFG.state, max_iter = 5000)
        model.fit(Xtr, ytr)

        dev_preds  = model.predict_proba(Xdev)[:,1]
        test_preds = test_preds + (model.predict_proba(mdl_preds)[:,1] / CFG.n_splits)
        score      = utils.ScoreMetric(ydev, dev_preds)
        PrintColor(f"---> Fold{fold_nb} Score = {score:.6f}", color = Fore.CYAN)
        scores = scores + (score / CFG.n_splits)

        ens_preds[dev_idx] = dev_preds

    PrintColor(f"\n---> Mean Score = {scores:.6f}")


def main():
    Path('figs').mkdir(exist_ok=True, parents=True)
    PrintColor(f"\n---> Configuration done!\n")
    Xtrain, Xtest, ytrain, ygrp = data_transform()
    Mdl_Master = model_training()
    single_model(Mdl_Master, Xtrain, Xtest, ytrain, ygrp)
    ensemble()

if __name__=="__main__":
    main()
