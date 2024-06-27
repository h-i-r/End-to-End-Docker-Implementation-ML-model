import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
warnings.filterwarnings("ignore")


def preliminary_cleaning(df):
    '''
    This function performs the following tasks:
    - adjust column datatypes
    - ensure data quality
    - create custom column 'dormancy_days'
    - add higher order terms (income) for linearity assumption
    - drop unnecessary columns
    '''

    # change dtypes & drop bugs
    df.consultant_id = df.consultant_id.astype('Int64')
    df.count_phone_calls = df.count_phone_calls.astype('Int64')
    df = df.loc[~(df['gender'] == 'company')]

    # custom columns
    df['registered_at'] = pd.to_datetime(df['registered_at'])
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['dormancy_days'] = (df['created_at'] - df['registered_at']).dt.days.astype('int')

    # drop
    df.drop(columns=['created_at', 'assign_at', 'complete_at', 'lost_at', 'registered_at', 'consultant_id'], inplace=True)

    return df

def drop_sparse_column(df, threshold):
    '''
    This function performs the following tasks:
    - remove features which are empty based on the set threshold
    '''

    missing_percentage = (df.isna().sum()) / (len(df)) * 100
    sparse_columns = missing_percentage[missing_percentage >= threshold].index
    print('dropping... ', sparse_columns)
    df = df.drop(columns=sparse_columns)
    return df


def impute_missing_values(df):
    '''
    This function performs the following tasks:
    - replace all missing values in categorical columns with 'Unknown'
    - replace all missing values in numerical columns with 0
    - perform custom category-based & grouped-based imputation for specific columns
    - engineer new features based on existing columns
    '''
    for column in df.columns:

        # categorical
        if df[column].dtype == 'object':
            if column in ['family_status', 'kids']:
                df[column].fillna('Unknown', inplace=True)

            elif column == 'city':
                top_5 = df['city'].value_counts().head(5).index
                df.loc[~df['city'].isin(top_5), 'city'] = 'Others'
                df['city'].fillna('Others', inplace=True)

            elif column == 'category_name':
                top_7 = df['category_name'].value_counts().head(7).index
                df.loc[~df['category_name'].isin(top_7), 'category_name'] = 'Others'
                df['category_name'].fillna('Others', inplace=True)

            elif column == 'mkt_channel':
                top_8 = df['mkt_channel'].value_counts().head(8).index
                df.loc[~df['mkt_channel'].isin(top_8), 'mkt_channel'] = 'Others'
                df['mkt_channel'].fillna('Others', inplace=True)

            elif column in ['gender']:
                df['gender'] = df.groupby(['employment', 'life_aspect'])['gender'].transform(
                    lambda x: x.fillna(x.mode()[0]))

            else:
                df[column].fillna('0', inplace=True)


        # numerical
        elif df[column].dtype != 'object':

            if column == 'count_phone_calls':
                df['count_phone_calls'] = df.groupby(['life_aspect', 'mkt_channel'])['count_phone_calls'].transform(
                    lambda x: x.fillna(x.mode()[0])).astype('int')

            elif column in ['income']:
                df['income'] = df.groupby(['employment'])['income'].transform(lambda x: x.fillna(x.median())).astype(
                    'float')

            elif column in ['age']:
                df['age'] = df.groupby(['employment', 'life_aspect'])['age'].transform(lambda x: x.fillna(x.mode()[0]))
                bins = [0, 30, 45, 55, float('inf')]
                labels = ['0-30', '31-45', '46-55', '56+']
                df['age'] = pd.cut(df['age'], bins=bins, labels=labels, right=False).astype('str')

            elif column == 'customer_id':
                deals_per_customer = df['customer_id'].value_counts()
                df['deals_per_customer'] = df['customer_id'].map(deals_per_customer)
                df.drop(columns=['customer_id'], inplace=True)
            else:
                df[column].fillna(df[column].median(), inplace=True)

    return df


def create_binary_indicator(df):
    '''
    This function performs the following tasks:
    - create binary indicator for regression
    '''

    # binary indicator
    df = df.loc[(df.state == 'completed') | ((df.state == 'lost'))]
    df['state'] = df['state'].replace({'completed': 1, 'lost': 0})
    y = df['state']

    return df, y


def encode(df):
    '''
     This function performs the following tasks:
     - separate id & state column before encoding
     - group columns into categorical & numerical for ColumnTransformation
     - perform respective transformations and merge id & state columns back into the df to plot correlation etc.
    '''

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse=False, drop='first')

    id_column = df[['id', 'state']]
    df.drop(columns=['id', 'state'], inplace=True)

    categorical_col = df.select_dtypes(include=['object']).columns
    numerical_col = df.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_col),
            ('cat', categorical_transformer, categorical_col)])

    preprocessed_data = preprocessor.fit_transform(df)
    cat_column_names = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_col)
    preprocessed_column_names = np.concatenate((numerical_col, cat_column_names))

    df = pd.concat(
        [id_column.reset_index(drop=True), pd.DataFrame(preprocessed_data, columns=preprocessed_column_names)], axis=1)

    return df
