import pandas as pd
from tabpfn import TabPFNRegressor
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

class Model:

    def __init__(self, catboost_params=None, cat_cols=None):
        if catboost_params is None:
            self.catboost_params = {'iterations': 2500,
                                    'learning_rate': 0.061824808908317305,
                                    'depth': 10,
                                    'l2_leaf_reg': 6.111025595991382,
                                    'random_strength': 7.1230012835574765,
                                    'loss_function': 'MAPE',
                                    'eval_metric': 'MAPE',
                                    'early_stopping_rounds': 100}
        else:
            self.catboost_params = catboost_params
        
        if cat_cols is None:
            self.cat_cols = ['transport_type', 'origin_kma', 'destination_kma']
        else:
            self.cat_cols = cat_cols

        self.catboost_regressor = CatBoostRegressor(**self.catboost_params)


    def fit_predict(self, X_train, y_train, X_valid):

        #check if the X_train and y_train are the same length
        assert len(X_train) == len(y_train), "X_train and y_train are not the same length"

        #check if X_train and X_valid are pandas dataframes and y_train is a pandas series
        assert isinstance(X_train, pd.DataFrame), "X_train is not a pandas dataframe"
        assert isinstance(X_valid, pd.DataFrame), "X_valid is not a pandas dataframe"
        assert isinstance(y_train, pd.Series), "y_train is not a pandas series"

        #check that index of X_train and y_train are the same
        assert X_train.index.equals(y_train.index), "Index of X_train and y_train are not the same"

        #check if the columns in X_train and X_valid are the same and self.cat_cols are in X_train
        assert X_train.columns.equals(X_valid.columns), "X_train and X_valid columns are not the same"
        assert all(col in X_train.columns for col in self.cat_cols), "Not all categorical columns are in X_train"

        #check that index of X_train and y_train are the same
        assert X_train.index.equals(y_train.index), "Index of X_train and y_train are not the same"

        #Fit CatBoost and get predictions
        print("Fitting CatBoost...")
        train_pool = Pool(X_train, y_train, cat_features=self.cat_cols)
        self.catboost_regressor.fit(train_pool, early_stopping_rounds=100, verbose=200)
        catboost_pred = self.catboost_regressor.predict(X_valid)

        print("Getting TabPFN predictions...")
        #Get TabPFN predictions. Don't need to train TabPFN as it is "in-context" learning
        grouped_valid = X_valid.groupby(["origin_kma", "destination_kma"])
        tabpfn_pred = pd.Series(index=X_valid.index, dtype=float)

        # Loop over each group => single (origin, destination) pair
        for (origin, destination), group_df in tqdm(grouped_valid):
            #Grab the suitable training subset
            train_subset = self.__get_suitable_train_subset__(X_train, origin, destination, max_rows=5000)
            y_subset = y_train.loc[train_subset.index]

            #Get predictions for this group
            reg = TabPFNRegressor()
            reg.fit(train_subset, y_subset)
            group_pred = reg.predict(group_df)

            #Store predictions in 'predictions' for these specific indices
            tabpfn_pred.loc[group_df.index] = group_pred

        blended_pred = 0.25 * catboost_pred + 0.75 * tabpfn_pred

        return blended_pred


    def __get_suitable_train_subset__(self, train_df, origin, destination, max_rows=5000):
        """
        Returns a subset of train_df up to 'max_rows':
        1) All rows with the same (origin, destination).
        2) If < max_rows, fill with the most recent rows from the same origin
        but different destinations, until you reach max_rows.
        """
        # A) All with same (origin, destination)
        same_ori_dst = train_df[
            (train_df['origin_kma'] == origin) &
            (train_df['destination_kma'] == destination)
        ]
        
        if len(same_ori_dst) >= max_rows:
            # If enough rows, just pick the most recent 'max_rows'
            subset = same_ori_dst.tail(max_rows)
        else:
            # Use all same (origin, destination)
            subset = same_ori_dst
            
            # B) Fill up with same origin, other destinations
            remainder = max_rows - len(subset)
            same_origin_other_dst = train_df[
                (train_df['origin_kma'] == origin) &
                (train_df['destination_kma'] != destination)
            ].sort_values(['year', 'month', 'day', 'hour'])  # ensure chronological
            
            additional = same_origin_other_dst.tail(remainder)
            subset = pd.concat([subset, additional])
        
        return subset