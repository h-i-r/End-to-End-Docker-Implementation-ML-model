import warnings
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from plots import plot_auc_roc, plot_features_for_random_forest, plot_qq, plot_lasso_coef, \
    plot_accuracies, plot_f1_scores, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")


def random_forest_model(df, y, estimators, show_plots):
    '''
      This function performs the following tasks:
      - under sample for class imbalance
      - train random forest
      - ROC curve
      - confusion matrix
      - feature importance

    '''

    # Drop state label feature
    df.drop(columns=['state'], inplace=True)

    X_sample, X_test, X_train, X_val, y_sample, y_test, y_train, y_val = split_data(df, y)

    train_accuracies = []
    test_accuracies = []
    val_accuracies = []

    train_f1_scores = []
    test_f1_scores = []
    val_f1_scores = []

    for estimator in estimators:
        # train
        clf = RandomForestClassifier(n_estimators=estimator, class_weight='balanced')
        clf = clf.fit(X_train, y_train)
        y_pred_clf = clf.predict(X_test)

        # mean cross-validation score on training data
        train_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
        test_f1_score = f1_score(y_test, y_pred_clf)

        print(f'n_estimators = {estimator}, Mean Score on 3-Fold CV: {train_scores.mean():.3f}')
        print(f'n_estimators = {estimator}, Test F1-Score : {test_f1_score}')

        #  ROC curve
        plot_auc_roc(X_train, y_train, X_test, y_test, X_val, y_val, clf, show_plots, "random_forest", estimator)

        # residuals & QQ
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        plot_qq(y_test, y_pred_proba, show_plots, "random_forest", estimator)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_clf).ravel()
        print("Random Forest Classifier Accuracy\t", accuracy_score(y_test, y_pred_clf))
        print('\n	~~~~ Confusion Matrix of Random Forest Classifier ~~~~~')
        print('		TP:', tp, '		TN:', tn)
        print('		FP:', fp, '			FN:', fn)
        print(classification_report(y_test, y_pred_clf))

        # Accuracies
        train_pred, train_accuracies = calculate_accuracy_scores(X_train, y_train, clf, train_accuracies)
        test_pred, test_accuracies = calculate_accuracy_scores(X_test, y_test, clf, test_accuracies)
        val_pred, val_accuracies = calculate_accuracy_scores(X_val, y_val, clf, val_accuracies)

        # F1-Scores
        train_f1_scores = calculate_f1_scores(y_train, train_pred, train_f1_scores)
        test_f1_scores = calculate_f1_scores(y_test, test_pred, test_f1_scores)
        val_f1_scores = calculate_f1_scores(y_val, val_pred, val_f1_scores)

    # accuracies against flexibility
    plot_accuracies(estimators, test_accuracies, train_accuracies, val_accuracies, "random_forest")

    # f1-score
    plot_f1_scores(estimators, test_f1_scores, train_f1_scores, val_f1_scores, "random_forest")

    # Confusion Matrix plot
    plot_confusion_matrix(y_pred_clf, y_test, "random_forest")

    # feature importance
    plot_features_for_random_forest(X_train, clf, show_plots)

    # pickle model
    random_forest_model_object = 'models/random_forest.pkl'
    pickle.dump(clf, open(random_forest_model_object, 'wb'))

    return X_sample, y_sample, clf


def logistic_regression_model(df, y, lasso_penalty, show_plots):
    '''
      This function performs the following tasks:
      - under sample for class imbalance
      - train logistic regression
      - ROC curve
      - confusion matrix
    '''

    # Drop state label feature
    df.drop(columns=['state'], inplace=True)

    X_sample, X_test, X_train, X_val, y_sample, y_test, y_train, y_val = split_data(df, y)

    train_accuracies = []
    test_accuracies = []
    val_accuracies = []

    train_f1_scores = []
    test_f1_scores = []
    val_f1_scores = []

    penalty = lasso_penalty

    for penalty_value in penalty:
        clf = LogisticRegression(C=penalty_value, penalty='l1', solver='liblinear')  # lasso
        # clf = LogisticRegression(C=penalty_value, class_weight='balanced') # ridge
        clf.fit(X_train, y_train)
        y_pred_clf = clf.predict(X_test)

        # residuals & QQ
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        plot_qq(y_test, y_pred_proba, show_plots, "logistic_regression", penalty_value)

        # mean roc & f1-score on 3-Fold
        train_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
        test_f1_score = f1_score(y_test, y_pred_clf)

        # ROC curve
        print(f'\nC parameter = {penalty_value}, Train Mean ROC AUC: {train_scores.mean():.3f}')
        print(f'\nC parameter = {penalty_value}, Test F1-Score: {test_f1_score}\n')

        plot_auc_roc(X_train, y_train, X_test, y_test, X_val, y_val, clf, show_plots, "logistic_regression",
                     penalty_value)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_clf).ravel()
        print("\nBinary Logistic Regression Classifier Accuracy: \n", accuracy_score(y_test, y_pred_clf))
        print('\n	~~~~Confusion Matrix of Binary Logistic Regression Classifier ~~~~~')
        print('		TP:', tp, '		TN:', tn)
        print('		FP:', fp, '			FN:', fn)
        print(classification_report(y_test, y_pred_clf))

        # Accuracies
        train_pred, train_accuracies = calculate_accuracy_scores(X_train, y_train, clf, train_accuracies)
        test_pred, test_accuracies = calculate_accuracy_scores(X_test, y_test, clf, test_accuracies)
        val_pred, val_accuracies = calculate_accuracy_scores(X_val, y_val, clf, val_accuracies)

        # F1-Scores
        train_f1_scores = calculate_f1_scores(y_train, train_pred, train_f1_scores)
        test_f1_scores = calculate_f1_scores(y_test, test_pred, test_f1_scores)
        val_f1_scores = calculate_f1_scores(y_val, val_pred, val_f1_scores)

    # Plot train, test, and validation accuracies against flexibility
    plot_accuracies(penalty, test_accuracies, train_accuracies, val_accuracies, "logistic_regression")
    # Plot F1
    plot_f1_scores(penalty, train_f1_scores, test_f1_scores, val_f1_scores, "logistic_regression")
    # Confusion Matrix plot
    plot_confusion_matrix(y_pred_clf, y_test, "logistic_regression")
    # Lasso coefficients
    plot_lasso_coef(X_test, clf, show_plots)

    # pickle model
    lr_object = 'models/logistic_regression.pkl'
    pickle.dump(clf, open(lr_object, 'wb'))

    return X_sample, y_sample, clf


def split_data(df, y):
    undersampler = RandomUnderSampler(random_state=42)
    df, y = undersampler.fit_resample(df, y)

    X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.6, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
    X_sample, X_val, y_sample, y_val = train_test_split(X_val, y_val, test_size=0.50, random_state=42)
    X_train = X_train.drop(columns=['id'])
    X_test = X_test.drop(columns=['id'])
    X_val = X_val.drop(columns=['id'])

    return X_sample, X_test, X_train, X_val, y_sample, y_test, y_train, y_val


def calculate_accuracy_scores(X, y, clf, accuracies):
    predictions = clf.predict(X)
    accuracy = accuracy_score(y, predictions)
    accuracies.append(accuracy)
    return predictions, accuracies


def calculate_f1_scores(y, pred, f1_scores_list):
    f1_score_value = f1_score(y, pred)
    f1_scores_list.append(f1_score_value)
    return f1_scores_list
