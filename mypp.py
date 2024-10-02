from gc import collect
import pandas as pd
import os
from config import *
from myimports import *

def save_fig_path(directory='figs'):
    # Specify the directory where you want to save the figures
    # Get all the files in the directory
    files = os.listdir(directory)

    # Filter out files that have a numeric filename and a valid image extension (e.g., .png or .jpg)
    image_files = [f for f in files if f.split('.')[-1].lower() in ['png', 'jpg'] and f.split('.')[0].isdigit()]

    # Extract the numeric part of the filenames and find the maximum
    if image_files:
        max_num = max([int(f.split('.')[0]) for f in image_files])
    else:
        max_num = 0  # If no files, start from 0

    # Define the new filename by incrementing the max number
    new_filename = f'{max_num + 1}.png'
    new_filepath = os.path.join(directory, new_filename)
    return new_filepath

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets
    2. In this case, no need to process the original data
    3. Check information and description
    4. Check unique values and nulls
    5. Collate starting features
    6. Create the cross-validation folds as a column for all future runs
    """;

    def __init__(self):
        self.train             = pd.read_csv(os.path.join(CFG.ip_path,"train.csv"), index_col = 'id')
        self.test              = pd.read_csv(os.path.join(CFG.ip_path ,"test.csv"), index_col = 'id')
        self.target            = CFG.target
        self.conjoin_orig_data = True if CFG.nb_orig > 0 else False
        self.dtl_preproc_req   = CFG.dtl_preproc_req
        self.test_req          = CFG.test_req

        self.original = pd.read_csv(CFG.orig_path)
        self.original.index.name = "id"
        self.original.index = range(len(self.original))
        self.original = self.original[self.train.columns]

        self.sub_fl = pd.read_csv(os.path.join(CFG.ip_path, "sample_submission.csv"))
        PrintColor(f"Data shapes - train-test-original | {self.train.shape} {self.test.shape} {self.original.shape}")

        for tbl in [self.train, self.original, self.test]:
            obj_cols      = tbl.select_dtypes(include = ["object", "category"]).columns
            tbl.columns   = tbl.columns.str.replace(r"\(|\)|\.|\s+","", regex = True)

    def _VisualizeDF(self):
        "This method visualizes the heads for the train, test and original data"

        PrintColor(f"\nTrain set head", color = Fore.CYAN)
        # display(self.train.head(5).style.format(precision = 3))

        PrintColor(f"\nTest set head", color = Fore.CYAN)
        # display(self.test.head(5).style.format(precision = 3))

        PrintColor(f"\nOriginal set head", color = Fore.CYAN)
        # display(self.original.head(5).style.format(precision = 3))

    def _AddSourceCol(self):
        self.train['Source']    = "Competition";
        self.test['Source']     = "Competition";
        self.original['Source'] = 'Original';

        self.strt_ftre = self.test.columns;
        return self;

    def _CollateInfoDesc(self):
        if self.dtl_preproc_req == "Y":
            PrintColor(f"\n{'-' * 20} Information and description {'-' * 20}\n", color = Fore.MAGENTA);

            # Creating dataset information and description:
            for lbl, df in {'Train': self.train, 'Test': self.test, 'Original': self.original}.items():
                PrintColor(f"\n{lbl} description\n");
                # display(df.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\
                #         transpose().\
                #         drop(columns = ['count'], errors = 'ignore').\
                #         drop([self.target], axis=0, errors = 'ignore').\
                #         style.format(formatter = '{:,.2f}').\
                #         background_gradient(cmap = 'Blues')
                #        );

                PrintColor(f"\n{lbl} information\n");
                # display(df.info());
                collect();
        return self;

    def _CollateUnqNull(self):

        if self.dtl_preproc_req == "Y":
            # Dislaying the unique values across train-test-original:-
            PrintColor(f"\nUnique and null values\n")
            _ = pd.concat([self.train[self.strt_ftre].nunique(),
                           self.test[self.strt_ftre].nunique(),
                           self.original[self.strt_ftre].nunique(),
                           self.train[self.strt_ftre].isna().sum(axis=0),
                           self.test[self.strt_ftre].isna().sum(axis=0),
                           self.original[self.strt_ftre].isna().sum(axis=0)
                          ],
                          axis=1)
            _.columns = ['Train_Nunq', 'Test_Nunq', 'Original_Nunq',
                         'Train_Nulls', 'Test_Nulls', 'Original_Nulls'
                        ]
            # display(_.T.style.background_gradient(cmap = 'Blues', axis=1).\
            #         format(formatter = '{:,.0f}')
            #        )

        return self;

    def _ConjoinTrainOrig(self):
        if self.conjoin_orig_data :
            PrintColor(f"\n\nTrain shape before conjoining with original = {self.train.shape}")
            train = pd.concat([self.train, self.original], axis=0, ignore_index = True)
            PrintColor(f"Train shape after conjoining with original= {train.shape}")

            train.index = range(len(train))
            train.index.name = 'id'

        else:
            PrintColor(f"\nWe are using the competition training data only")
            train = self.train
        return train

    def DoPreprocessing(self):
        self._VisualizeDF()
        self._AddSourceCol()
        self._CollateInfoDesc()
        self._CollateUnqNull()
        self.train = self._ConjoinTrainOrig()
        self.train.index = range(len(self.train))

        self.cat_cols  = list(self.test.drop("Source", axis=1).select_dtypes("object").columns)
        self.cont_cols = [c for c in self.strt_ftre if c not in self.cat_cols + ['Source']]
        return self

collect();
print();

class FeaturePlotter:
    """
    This class develops plots for the targets, continuous and category features
    """;

    def __init__(
        self, target: str, ftre_plots_req: bool, title_specs : dict, grid_specs: dict,
    ):
        self.target         = target
        self.ftre_plots_req = ftre_plots_req
        self.title_specs    = title_specs
        self.grid_specs     = grid_specs

    def MakeTgtPlot(
        self, train, original
    ):
        "This method returns the target plots";

        if self.ftre_plots_req == True:
            fig, axes = \
            plt.subplots(
                1,2, figsize = (14, 4), sharey = True,
                gridspec_kw = {'wspace': 0.35}
            );

            for i, df in tqdm(enumerate([train, original]), f"Target plot- {self.target} ---> "):
                ax= axes[i];

                a = df[self.target].value_counts(normalize = True);
                a.sort_index().plot.bar(color = 'tab:blue', ax = ax);
                df_name = 'Train' if i == 0 else "Original";
                _ = ax.set_title(f"\n{df_name} data- {self.target}\n", **self.title_specs);
                ax.set_yticks(
                    np.arange(0,1.01, 0.05),
                    labels = np.around(np.arange(0,1.01, 0.05), 2),
                    fontsize = 7.0
                )

            plt.tight_layout();
            fig.savefig(save_fig_path('figs'))

            # plt.show();

    def MakeCatFtrePlots(
        self, cat_cols, train, test, original
    ):
        "This method returns the category feature plots";

        if cat_cols != [] and self.ftre_plots_req == True:
            fig, axes = \
            plt.subplots(len(cat_cols), 3,
                         figsize = (25, len(cat_cols)* 4.5),
                         gridspec_kw = {'wspace': 0.45, 'hspace': 0.40},
                        );

            for i, col in enumerate(cat_cols):
                ax = axes[i, 0] if len(cat_cols) > 1 else axes[0]
                a = train[col].value_counts(normalize = True)
                a.sort_index().plot.barh(ax = ax, color = '#007399')
                ax.set_title(f"{col}_Train", **self.title_specs)
                ax.set_xticks(np.arange(0.0, 1.01, 0.05),
                              labels = np.round(np.arange(0.0, 1.01, 0.05),2),
                              rotation = 90
                             );
                ax.set(xlabel = '', ylabel = '')
                del a;

                ax = axes[i, 1] if len(cat_cols) > 1 else axes[1];
                a = test[col].value_counts(normalize = True);
                a.sort_index().plot.barh(ax = ax, color = '#0088cc');
                ax.set_title(f"{col}_Test", **self.title_specs);
                ax.set_xticks(np.arange(0.0, 1.01, 0.05),
                              labels = np.round(np.arange(0.0, 1.01, 0.05),2),
                              rotation = 90
                             );
                ax.set(xlabel = '', ylabel = '');
                del a;

                ax = axes[i, 2] if len(cat_cols) > 1 else axes[2];
                a = original[col].value_counts(normalize = True);
                a.sort_index().plot.barh(ax = ax, color = '#0047b3');
                ax.set_title(f"{col}_Original", **self.title_specs);
                ax.set_xticks(np.arange(0.0, 1.01, 0.05),
                              labels = np.round(np.arange(0.0, 1.01, 0.05), 2),
                              rotation = 90
                             );
                ax.set(xlabel = '', ylabel = '');
                del a;

            plt.suptitle(f"Category column plots", **self.title_specs, y= 0.96);
            plt.tight_layout();
            fig.savefig(save_fig_path('figs'))
            # plt.show();

    def MakeContColPlots(
        self, cont_cols, train, test, original,
    ):
        "This method returns the continuous feature plots";

        if self.ftre_plots_req == True:
            df = pd.concat([train[cont_cols].assign(Source = 'Train'),
                            test[cont_cols].assign(Source = 'Test'),
                            original[cont_cols].assign(Source = "Original")
                           ],
                           axis=0, ignore_index = True
                          );

            fig, axes = plt.subplots(len(cont_cols), 4 ,figsize = (16, len(cont_cols) * 4.2),
                                     gridspec_kw = {'hspace': 0.35,
                                                    'wspace': 0.3,
                                                    'width_ratios': [0.80, 0.20, 0.20, 0.20]
                                                   }
                                    );

            for i,col in enumerate(cont_cols):
                ax = axes[i,0];
                sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source',
                            palette = ['#0039e6', '#ff5500', '#00b300'],
                            ax = ax, linewidth = 2.1
                           );
                ax.set_title(f"\n{col}", **self.title_specs);
                ax.grid(**self.grid_specs);
                ax.set(xlabel = '', ylabel = '');

                ax = axes[i,1];
                sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25,
                            color = '#33ccff', saturation = 0.90, linewidth = 0.90,
                            fliersize= 2.25,
                            ax = ax
                           )
                ax.set(xlabel = '', ylabel = '');
                ax.set_title(f"Train", **self.title_specs);

                ax = axes[i,2];
                sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, fliersize= 2.25,
                            color = '#80ffff', saturation = 0.6, linewidth = 0.90,
                            ax = ax);
                ax.set(xlabel = '', ylabel = '');
                ax.set_title(f"Test", **self.title_specs);

                ax = axes[i,3];
                sns.boxplot(data = df.loc[df.Source == 'Original', [col]], y = col, width = 0.25, fliersize= 2.25,
                            color = '#99ddff', saturation = 0.6, linewidth = 0.90,
                            ax = ax);
                ax.set(xlabel = '', ylabel = '');
                ax.set_title(f"Original", **self.title_specs);

            plt.suptitle(f"\nDistribution analysis- continuous columns\n", **CFG.title_specs,
                         y = 0.95, x = 0.50
                        );
            plt.tight_layout();
            fig.savefig(save_fig_path('figs'))
            # plt.show();

    def CalcSkew(self, cont_cols, train, test, original):
        "This method calculates the skewness across columns";

        if self.ftre_plots_req == True:
            skew_df = pd.DataFrame(index = cont_cols);
            for col, df in {"Train"   : train[cont_cols],
                            "Test"    : test[cont_cols],
                            "Original": original[cont_cols]
                           }.items():
                skew_df = \
                pd.concat([skew_df,
                           df.drop(columns =  [self.target, "Source", "id"], errors = "ignore").skew()],
                           axis=1).rename({0: col}, axis=1);

            PrintColor(f"\nSkewness across independent features\n");
            # display(skew_df.transpose().style.format(precision = 2).background_gradient("PuBuGn"));

print();
collect();
