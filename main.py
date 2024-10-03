import pandas as pd
from myimports import *
from training import *
from mypp import *
from gc import collect
from pathlib import Path
import wandb
import random
import optuna
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf

class CFG:
    pass

def config_init(cfg: DictConfig) -> None:
    # Print the configuration in YAML format (optional, for debugging)
    # print(OmegaConf.to_yaml(cfg))

    # Load parameters from 'general' section of the config to class attributes
    for key, value in cfg['general'].items():
        setattr(CFG, key, value)

# with open('config.yaml', 'r') as file:
    # config = yaml.safe_load(file)

def wandb_init(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))  # For debugging, print the config in YAML format
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Convert DictConfig to a dictionary

    # Initialize wandb with the converted dictionary
    wandb.init(
        project=cfg_dict['wandb_params']['project'],
        config=cfg_dict,  # Now passing the standard dictionary
        mode=cfg_dict['wandb_params']['mode']
    )

# @hydra.main(version_base=None, config_path=".", config_name="config")
def model_training_params(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Convert DictConfig to a dictionary
    models_config = cfg_dict['models']

    # Use the loaded config to initialize models
    Mdl_Master = {
        'LGBM1C': LGBMC(**models_config['LGBM1C']),
        'LGBM2C': LGBMC(**models_config['LGBM2C']),
        'XGB1C': XGBC(**models_config['XGB1C']),
        'CB1C': CBC(**models_config['CB1C']),
    }

    # Modify device parameter dynamically based on CFG.gpu_switch
    if CFG.gpu_switch == "ON":
        Mdl_Master['LGBM1C'].set_params(device='gpu')
        Mdl_Master['LGBM2C'].set_params(device='gpu')
        Mdl_Master['XGB1C'].set_params(device='cuda')
    else:
        Mdl_Master['LGBM1C'].set_params(device='cpu')
        Mdl_Master['LGBM2C'].set_params(device='cpu')
        Mdl_Master['XGB1C'].set_params(device='cpu')
    # print([64]*20)
    # print(type(Mdl_Master), Mdl_Master)
    # Return the model configuration dictionary
    return Mdl_Master



    # wandb.log({"acc": acc, "loss": loss})

def data_transform():

    ytrain = pp.train[CFG.target]
    Xtrain = pp.train.drop([CFG.target, CFG.grouper], axis= 1, errors = "ignore")
    PrintColor(f"---> Shapes = {Xtrain.shape} {ytrain.shape} {pp.test.shape}")


    Xtrain = MakeFtre(Xtrain, cat_mdl_cols)
    Xtest  = MakeFtre(pp.test, cat_mdl_cols)

    PrintColor(f"---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape}")

    cv_selector = \
    {
     "RKF"   : RKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
     "RSKF"  : RSKF(n_splits = CFG.n_splits, n_repeats= CFG.n_repeats, random_state= CFG.state),
     "SKF"   : SKF(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
     "KF"    : KFold(n_splits = CFG.n_splits, shuffle = True, random_state= CFG.state),
     "GKF"   : GKF(n_splits = CFG.n_splits)
    }
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

    # df['age_income_interaction'] = df['person_age'] * df['person_income']
    # df['income_loan_amnt_interaction'] = df['person_income'] * df['loan_amnt']
    # df['loan_amnt_int_rate_interaction'] = df['loan_amnt'] * df['loan_int_rate']
    # df['int_rate_to_income_ratio'] = df['loan_int_rate'] / df['person_income']
    # df['emp_length_to_age_ratio'] = df['person_emp_length'] / df['person_age']


    df[cat_cols] = df[cat_cols].astype("category")
    return df


def barplot_scores(scores_all):
    data = []
    for model, scores in scores_all.items():
        data.append([model, -np.log10(1-scores[0]), -np.log10(1-scores[1])])
    table = wandb.Table(data=data, columns=["Model", "OOF Mean", "Train Mean"])
    wandb.log({"Model Performance": table})
    # Use wandb built-in bar plot functionality
    wandb.log({
        "OOF Mean Values": wandb.plot.bar(table, "Model", "OOF Mean", title="OOF Mean Values"),
        "Train Mean Values": wandb.plot.bar(table, "Model", "Train Mean", title="Train Mean Values")
    })



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
    scores_all = {}
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

        fitted_models, oof_preds, test_preds, ftreimp, mdl_best_iter, scores_all[method] =  \
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

        # del fitted_models, oof_preds, test_preds, ftreimp, sel_mdl_cols
        # print()
        # collect();
    barplot_scores(scores_all)
    return OOF_Preds, Mdl_Preds
    # _ = utils.CleanMemory();


def ensemble(Xtrain, ytrain, ygrp, OOF_Preds, Mdl_Preds):
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
    wandb.log({"logistic_ensemble": scores})

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg:DictConfig):
    config_init(cfg)
    wandb_init(cfg)
    Path('figs').mkdir(exist_ok=True, parents=True)
    PrintColor(f"\n---> Configuration done!\n")
    Xtrain, Xtest, ytrain, ygrp = data_transform()
    Mdl_Master = model_training_params(cfg)
    OOF_Preds, Mdl_Preds = single_model(Mdl_Master, Xtrain, Xtest, ytrain, ygrp)
    ensemble(Xtrain, ytrain, ygrp, OOF_Preds, Mdl_Preds)


if __name__=="__main__":

    # collect()
    pp = Preprocessor()
    pp.DoPreprocessing()
    cat_mdl_cols = \
        list(pp.test.select_dtypes([np.int8, np.int16, np.int32, np.int64]).columns) + pp.cat_cols
    main()
