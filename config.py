class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;

    # Data preparation:-
    version_nb  = 1
    model_id    = "V1_2"
    model_label = "ML"

    test_req           = False
    test_sample_frac   = 0.05

    gpu_switch         = "OFF"
    state              = 42
    target             = f"loan_status"
    grouper            = f""
    tgt_mapper         = {}

  #  ip_path            = f"/kaggle/input/playground-series-s4e10"
    ip_path = "data"
    op_path            = f"working/"
    orig_path          = f"data/credit_risk_dataset.csv"

    dtl_preproc_req    = True
    ftre_plots_req     = True
    ftre_imp_req       = True

    nb_orig            = 1
    orig_all_folds     = True

    # Model Training:-
    pstprcs_oof        = False
    pstprcs_train      = False
    pstprcs_test       = False
    ML                 = True
    test_preds_req     = False

    pseudo_lbl_req     = "N"
    pseudolbl_up       = 0.975
    pseudolbl_low      = 0.00

    n_splits           = 3 if test_req == True else 10
    n_repeats          = 1
    nbrnd_erly_stp     = 100
    mdlcv_mthd         = 'SKF'

    # Ensemble:-
    ensemble_req       = True
    optuna_req         = False
    metric_obj         = 'maximize'
    ntrials            = 10 if test_req == True else 300

    # Global variables for plotting:-
    grid_specs = {'visible'  : True,
                  'which'    : 'both',
                  'linestyle': '--',
                  'color'    : 'lightgrey',
                  'linewidth': 0.75
                 }

    title_specs = {'fontsize'   : 9,
                   'fontweight' : 'bold',
                   'color'      : '#992600',
                  }
