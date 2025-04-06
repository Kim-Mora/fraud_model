import pandas as pd
from pandas import DataFrame
from src.utils import calculate_distance, calculate_woes, \
                    split_train_test_set

class FraudFeatureEgineering:
    def __init__(self:any, df:DataFrame)-> None:
        self.df = df.copy()
        self.primary_key = 'credit_card_number'
        self.merchant_grouper = [self.primary_key, 'merchant']
        self.train_set = None
        self.test_set = None

    def create_time_delta_features(self:any, by_merchant:bool=False)-> None:
        """Create time delta features for a given credit card. It can be
           in the whole credit_card_history or grouped by merchant.

        Example:
        data = {credit_card_number: [1, 1, 1, 2, 2, 3, 3, 3, 3],
                merchant: [a, b, b, a ,a , a, c, c, a],
                timestamp: [1, 2, 3, 1, 2, 1, 2, 3, 2]}

        feature_result_for credit_card_number = 3:
            time_between_transactions = [nan, 1, 1, nan]
            time_between_Transactions_by_merchant = [nan, nan, nan, 1]
        
        Args:

            by_merchant (bool, optional): Switch to calcule feature by merchant. 
                                            Defaults to False.
        """
        grouper = self.primary_key
        col_name = 'time_between_transactions'
        if by_merchant:
            grouper = self.merchant_grouper
            col_name = 'time_between_transactions_by_merchant'
        self.df[col_name] = self.df.groupby(grouper)['timestamp'].diff()

    def get_running_statistic_features(self:any, col:str, mean:bool=True) -> None:
        """Gets for each transaction the avg or std for a given column
           for each merchant for each credit_card_number.

           Example:
            data = {credit_card_number: [1, 1, 1, 2, 2, 3, 3, 3, 3],
                    merchant: [a, b, b, a ,a , a, c, c, a],
                    amount: [100, 200, 300, 100, 200, 100, 200, 300, 200]}

            feature_result_for credit_card_number = 3:
                runing_mean_amount_by_merchant = [100, nan, nan, 100]

        Args:
            col (str): _description_
            mean (bool, optional): _description_. Defaults to True.
        """
        
        if mean:
            self.df[f'running_mean_{col}_by_merchant'] = \
                self.df.groupby(self.merchant_grouper)[col].expanding().mean().reset_index(
                    level=[0,1], drop=True
                )
        else:
            self.df[f'running_std_{col}_by_merchant'] = \
                self.df.groupby(self.merchant_grouper)[col].expanding().std().reset_index(
                    level=[0,1], drop=True
                )

    def get_is_anormal_amount_feature(self:any, confidence:int) -> None:
        """Calculate if the amount of the current transaction is beyond 
           the current mean amount plus a given confidence level on a given merchant:
           The confidence level is given as the follow:
           1: 90%
           2: 95%
           3: 99%

           Example:
            data = {credit_card_number: [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    merchant: [a, a, a, a ,a , a, a, a, a],
                    amount: [100, 200, 300, 100, 200, 100, 1500, 300, 200]}

            feature_result:
                is_anormal_mount = [0, 0, 0, 0, 0, 0, 1, 0, 0]


        Args:
            confidence (int): confidence level to calculate the feature
        """
        self.df['prev_running_mean'] = \
            self.df.groupby(self.merchant_grouper)['running_mean_Amount_by_merchant'].shift(periods=1)
        self.df['prev_running_std'] = \
            self.df.groupby(self.merchant_grouper)['running_std_Amount_by_merchant'].shift(periods=1)

        self.df['is_anormal_amount'] = (self.df['Amount'] > (self.df['prev_running_mean'] \
                                        + confidence * self.df['prev_running_std'])) * 1
        self.df.drop(['prev_running_mean', 'prev_running_std'], axis=1, inplace=True)

    def get_times_using_same_merchant_feature(self:any)-> None:
        """Get the number of times a given credit card buy on the same merchant.

           Example:
            data = {credit_card_number: [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    merchant: [a, b, c, d ,a , b, b, c, a]}

            feature_result:
                is_anormal_mount = [1, 1, 1, 1, 2, 2, 3, 3, 3]
        """

        self.df['times_using_same_merchant'] = (
            self.df.groupby(self.merchant_grouper).cumcount() + 1
        )

    def get_num_transactions_by_cc(self:any) -> None:
        """Get the cumulative count of transactions by user
           Example:
            data = {credit_card_number: [1, 2, 1, 2, 2, 2, 1],
                    merchant: [a, b, c, d ,a , b, b, c, a]}

            feature_result:
                is_anormal_mount = [1, 1, 2, 2, 3, 4, 5, 3]
        """
        self.df['num_transactions_by_cc'] = \
            self.df.groupby(self.primary_key)[self.primary_key].transform('count')

    def get_distance_between_application(self:any, by_merchant:bool=False) -> None:
        """Calculate the distance (in km) between two transactions of the same credit_card_number
           it can be both general or by merchant.
           

        Args:
            by_merchant (bool, optional): switch to calculate by merchant. Defaults to False.
        """
        col_name = 'distance_between_transactions'
        grouper = self.primary_key
        if by_merchant:
            col_name = 'distance_between_transactions_by_merchant'
            grouper = self.merchant_grouper
        self.df['prev_lat'] = self.df.groupby(grouper)['latitude'].shift(periods=1)
        self.df['prev_long'] = self.df.groupby(grouper)['longitude'].shift(periods=1)
        self.df[col_name] = self.df.apply(calculate_distance, axis=1)
        self.df.drop(['prev_lat', 'prev_long'], axis=1, inplace=True)

    def split_dataset_and_woe_encoding(self:any)-> tuple[DataFrame, DataFrame]:
        """Split the data into train and test set. Then it calculate the WoE
        encoding using only train data in order to prevent leakage. Finaly
        it transform the categorical column with the calculed woes. Aditionaly
        it saves the woe dict into a json file.

        Returns:
            tuple[DataFrame, DataFrame]: train_set and test_set
            with the transformed categorical feature.
        """
        
        self.train_set, self.test_set = split_train_test_set(self.df)
        woe_encoding = calculate_woes(self.train_set, 
                                      feature='merchant',
                                      target='Class')
        self.train_set['transformed_merchant'] = \
            self.train_set.merchant.map(woe_encoding)
        self.test_set['transformed_merchant'] = \
            self.test_set.merchant.map(woe_encoding)
        
    def get_fraud_features(self:any)->None:
        self.create_time_delta_features()
        self.create_time_delta_features(by_merchant=True)
        self.get_running_statistic_features(col='Amount')
        self.get_running_statistic_features(col='Amount', mean=False)
        self.get_is_anormal_amount_feature(confidence=2)
        self.get_times_using_same_merchant_feature()
        self.get_num_transactions_by_cc()
        self.get_distance_between_application()
        self.get_distance_between_application(by_merchant=True)
        self.split_dataset_and_woe_encoding()
        return self.train_set, self.test_set
