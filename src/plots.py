import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")


def plot_collinear_correlations(df, show_plots):
    '''
      This function performs the following tasks:
      - Plot correlation against 'state' indicator
      - Visualize correlations in Heatmap
      - check for potential multi-collinearity between features
    '''

    # Corr
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    state_corr_num = df[numerical_columns].corrwith(df['state'])
    plt.figure(figsize=(20, 20))
    state_corr_num.plot(kind='bar', color='skyblue')
    plt.title('Correlation of State Variable')
    plt.xlabel('Columns')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.savefig('plots/correlation_plots')
    if show_plots:
        plt.show()

    # Heatmap
    correlation_matrix = df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('plots/heatmap')
    if show_plots:
        plt.show()

    # Calculating collinearity
    correlation_matrix = df.corr().abs()
    collinear_features = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > 0.9:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                collinear_features.add((colname_i, colname_j))

    if len(collinear_features) > 0:
        print("Collinear Feature Pairs:")
        for pair in collinear_features:
            print(pair)

    # drop correlated variables
    df.drop(columns=['category_name_Statutory health insurance'], inplace=True)
    df.drop(columns=['employment_Others'], inplace=True)
    df.drop(columns=['kids_Unknown'], inplace=True)
    df.drop(columns=['income'], inplace=True)


def plot_auc_roc(X_train, y_train, X_test, y_test, X_val, y_val, clf, show_plots, model_type, hyperparameter):
    FP_train, TP_train, roc_train = calculate_roc(X_train, y_train, clf)

    FP_test, TP_test, roc_test = calculate_roc(X_test, y_test, clf)

    FP_val, TP_val, roc_val = calculate_roc(X_val, y_val, clf)

    plt.figure(figsize=(20, 20))
    plt.plot(FP_train, TP_train, color='blue', lw=2, label='Train ROC curve (AUC = %0.2f)' % roc_train)
    plt.plot(FP_test, TP_test, color='green', lw=2, label='Test ROC curve (AUC = %0.2f)' % roc_test)
    plt.plot(FP_val, TP_val, color='red', lw=2, label='Validation ROC curve (AUC = %0.2f)' % roc_val)

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_type} on hyperparameter: {hyperparameter}')
    plt.legend(loc='lower right')
    plt.savefig(f'plots/roc_{model_type}_{hyperparameter}.png')
    if show_plots:
        plt.show()


def plot_confusion_matrix(y_pred_clf, y_test, model_type):
    cm = confusion_matrix(y_test, y_pred_clf, normalize='pred')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['completed', 'lost'])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", values_format='.2f')
    plt.title('Confusion Matrix', fontsize=40)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{model_type}')


def plot_features_for_random_forest(X_train, clf, show_plots):
    features = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    features_good = features.tail(10)
    features_bad = features.head(10)

    f, ax = plt.subplots(1, 2, figsize=(20, 20))

    features_good.plot(kind='barh', cmap='Pastel1', ax=ax[0], fontsize=20)
    ax[0].set_title('10 Most Important Features', fontsize=40)

    features_bad.plot(kind='barh', cmap='Pastel1', ax=ax[1], fontsize=20)
    ax[1].set_title('10 Least Important Features',fontsize=40)

    plt.savefig('plots/feature_importance_random_forest')
    if show_plots:
        plt.show()


def calculate_roc(X, y, clf):
    y_predict_proba = clf.predict_proba(X)[:, 1]
    FP, TP, _ = roc_curve(y, y_predict_proba)
    roc = auc(FP, TP)
    return FP, TP, roc


def plot_qq(y_test, y_pred_proba, show_plots, model_type, hyperparameter):
    # residuals
    plt.clf()
    residuals = y_test - y_pred_proba
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    stats.probplot(standardized_residuals, dist="norm", plot=plt)
    plt.title(f'QQ Plots - {model_type} on hyperparameter: {hyperparameter}')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Standardized residuals')
    plt.savefig(f'plots/qq_plots_{model_type}_{hyperparameter}.png')
    if show_plots:
        plt.show()


def plot_lasso_coef(X_test, clf, show_plots):
    coef = clf.coef_.ravel()
    feature_names = np.array(X_test.columns)
    # Plot coefficients
    plt.figure(figsize=(20, 20))
    plt.barh(feature_names, coef, color='blue')
    plt.xlabel('Coefficient Values')
    plt.ylabel('Columns')
    plt.title('Lasso Coefficients')
    plt.grid(True)
    plt.savefig('plots/lasso_coef')
    if show_plots:
        plt.show()


def plot_accuracies(penalty, test_accuracies, train_accuracies, val_accuracies, model_type):
    plt.close("all")
    plt.plot(penalty, train_accuracies, label='Train Accuracy', color='red')
    plt.plot(penalty, test_accuracies, label='Test Accuracy', color='blue')
    plt.plot(penalty, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Flexibility')
    plt.ylabel('Accuracy')
    # plt.xscale('log')
    plt.title(f'Train, Test, and Validation Accuracy vs. Flexibility ({model_type})')
    plt.legend()
    plt.savefig(f'plots/accuracy_comparison_{model_type}')
    # plt.show()


def plot_f1_scores(penalty, test_f1, train_f1, val_f1, model_type):
    plt.close("all")
    plt.plot(penalty, train_f1, label='Train F1-Score', color='red')
    plt.plot(penalty, test_f1, label='Test F1-Score', color='blue')
    plt.plot(penalty, val_f1, label='Validation F1-Score', color='green')
    plt.xlabel('Flexibility')
    plt.ylabel('F1-Score')
    # plt.xscale('log')
    plt.title(f'Train, Test, and Validation F1-Score vs. Flexibility ({model_type})')
    plt.legend()
    plt.savefig(f'plots/f1_comparison_{model_type}')
    # plt.show()