import warnings
import pandas as pd
import argparse
from preprocessing import preliminary_cleaning, drop_sparse_column, impute_missing_values, \
    create_binary_indicator, encode
from ranking import compare_rankings
from model import random_forest_model, logistic_regression_model
from plots import plot_collinear_correlations
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_column_threshold', type=int, default=60)
    parser.add_argument('--n_estimators', nargs='+', type=int, default=[50])
    parser.add_argument('--lasso_penalty', nargs='+', type=float, default=[10])
    parser.add_argument('--number_of_potential_customers', type=int, default=20)
    parser.add_argument('--file_name', type=str, default="Customers_Dataset.csv")
    parser.add_argument('--model_type', type=str, default="logistic_regression")
    parser.add_argument('--show_plots', type=bool, default=False)

    args = parser.parse_args()
    file_name = args.file_name
    sparse_column_threshold = args.sparse_column_threshold
    model_type = args.model_type
    number_of_potential_customers = args.number_of_potential_customers
    n_estimators = args.n_estimators
    lasso_penalty = args.lasso_penalty
    show_plots = args.show_plots

    print(f'Running Purchase Predictor on the following Arguments:\n '
          f'file_name = {file_name} \n'
          f'sparse_column_threshold = {sparse_column_threshold} \n '
          f'number_of_potential_customers = {number_of_potential_customers} \n'
          f'show_plots = {show_plots} \n'
          f'model_type = {model_type}')

    if model_type == "logistic_regression":
        print(f'lasso_penalty = {lasso_penalty} \n')
    elif model_type == "random_forest":
        print(f'n_estimators = {n_estimators} \n')

    # Load Data
    file_path = f'datasets/{file_name}'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found on the path: '{file_path}'")


    # Initial Cleaning and Processing
    df = preliminary_cleaning(df)

    # Remove Sparse Data
    df = drop_sparse_column(df, sparse_column_threshold)

    # Impute Missing Values
    df = impute_missing_values(df)

    # Indicator
    df, y = create_binary_indicator(df)

    # Encode
    df = encode(df)

    # Plotting Collinearity & Correlations
    plot_collinear_correlations(df, show_plots)

    # Run Model
    if model_type == "logistic_regression":
        X_sample, y_sample, clf = logistic_regression_model(df, y, lasso_penalty, show_plots)
    elif model_type == "random_forest":
        X_sample, y_sample, clf = random_forest_model(df,y, n_estimators, show_plots)

    # Rank
    compare_rankings(X_sample, clf, file_path, number_of_potential_customers, model_type)


if __name__ == "__main__":
    main()