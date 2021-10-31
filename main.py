"""
* @author: Wael Khalil
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy.interpolate import make_interp_spline
from sklearn.svm import SVC
from sklearn import svm

def read_dataset(filename):
    '''
    This function reads in the dataset of the filename that is passed in.
    :param filename: A string representing the filename of the dataset we are
    working with.
    :return: A dataframe representing the dataset.
    '''

    df = pd.read_csv(filename, header=None)
    return df


def feature_ranking_chi_squared(dataset, header_included = False, label_included = False):
    '''
    This function performs the Chi-squared test on our dataset allowing to
    rank the features in it.
    :param dataset: A dataframe representing the dataset we are working with
    :return: A dataframe that contains two columns, one column showing the
    column number of the feature in the original dataset, and the ranking scores
    of that feature.
    '''

    array = dataset.values

    # splitting the features and the class label
    if not header_included:
        X = array[:, 1:]
        Y = array[:, 0]

    else:
        X = array[1:, 1:]
        Y = array[1:, 0]

    test = SelectKBest(score_func=mutual_info_regression, k=60)
    fit = test.fit(X, Y)

    np.set_printoptions(precision=3)
    features = fit.transform(X)

    # Summarize selected features
    #print(features[0:5, :])

    # checking if label is included with our dataset that is passed in
    if label_included:
        num_features = len(dataset.columns) - 1
    else:
        num_features = len(dataset.columns)

    column_names = []
    for i in range(1,num_features+1):
        column_names.append(f"Column_{i}")

    # creating a dataframe out of our ranked features
    ranked_features = {'Feature Column Number':column_names,'Feature Ranking':fit.scores_}
    ranked_features_df = pd.DataFrame(ranked_features)

    return ranked_features_df


def retrieve_top_k_features(ranked_features_df, k):
    """
    This function retrieves a dataframe with the top k features in our dataset.
    :param ranked_features_df: A dataframe with the ranking score of our
    features.
    :param k: An integer representing the number of top features to retrieve
    :return: A dataframe that contains top k features in descending order
    """

    # sorting our features then grabbing the top k rows
    top_k_features_ascending = perform_sort(ranked_features_df, "Feature Ranking").tail(k)

    # transforming each of the columns into lists
    feature_column_number_col_arr = top_k_features_ascending['Feature Column Number'].to_list()
    feature_ranking_col_arr = top_k_features_ascending["Feature Ranking"].to_list()

    # reverse our lists so we see the ranked features in descending
    feature_column_number_col_arr.reverse()
    feature_ranking_col_arr.reverse()

    # prepare our data and create a dataframe for it
    top_k_features = {'Feature Ranking':feature_ranking_col_arr, 'Feature Column Number':feature_column_number_col_arr}
    top_k_features_df = pd.DataFrame(top_k_features)

    return top_k_features_df


def two_way_merge_sort(arr: list, feature_column_number: list, comparisons: list,
                       swaps: list):
    '''
    This function performs a two-way merge sort on the data passed to it.
    :param arr: An array (list) representing the data processed from file
    :param feature_column_number: A list that would contain the transformed species
    column.
    :param comparisons: A list that would store the number of comparisons
    :param swaps: A list that would store the number of exchanges
    :return: This function does not have a return. It modifies the array
    passed to it as one of its arguments (which is a copy).
    '''

    if len(arr) > 1:

        # Finding the mid of the array
        mid = len(arr) // 2

        # Dividing the array elements into two sublists, left and right
        L = arr[:mid]
        R = arr[mid:]

        L_species = feature_column_number[:mid]
        R_species = feature_column_number[mid:]

        # Sorting the first half
        two_way_merge_sort(L, L_species, comparisons, swaps)

        # Sorting the second half
        two_way_merge_sort(R, R_species, comparisons, swaps)

        left_index = right_index = sorted_index = 0

        # Copy data to temp arrays L[] and R[]
        while left_index < len(L) and right_index < len(R):
            comparisons[0] += 1
            swaps[0] += 1
            if L[left_index] < R[right_index]:
                arr[sorted_index] = L[left_index]
                # this takes care of sorting the species column
                feature_column_number[sorted_index] = L_species[left_index]
                left_index += 1

            else:
                arr[sorted_index] = R[right_index]
                # this takes care of sorting the species column
                feature_column_number[sorted_index] = R_species[right_index]
                right_index += 1

            sorted_index += 1

        # Checking if any element was left
        while left_index < len(L):
            swaps[0] += 1
            arr[sorted_index] = L[left_index]
            # this takes care of sorting the species column
            feature_column_number[sorted_index] = L_species[left_index]
            left_index += 1
            sorted_index += 1

        while right_index < len(R):
            swaps[0] += 1
            arr[sorted_index] = R[right_index]
            # this takes care of sorting the species column
            feature_column_number[sorted_index] = R_species[right_index]
            right_index += 1
            sorted_index += 1


def perform_sort(dataset, column_name):
    '''
    This is the function that makes sure that the data passed in is transformed
    into a Python list object before it calls the sorting algorithm on our data.
    :param dataset: A dataframe representing the dataset we are working with
    :param column_name: A string representing the column name we will perform
    the sort on
    :return: A dataframe that will contain two columns, the sorted feature
    ranking column, and the feature column number column.
    '''

    feature_ranking_data = dataset[column_name]
    feature_ranking_array = np.array(feature_ranking_data).tolist()
    column_name_array = np.array(dataset["Feature Column Number"]).tolist()

    # allows us to count comparisons and swaps
    comparisons = [0]
    swaps = [0]

    # calling our mergesort function
    two_way_merge_sort(feature_ranking_array, column_name_array, comparisons, swaps)

    # creating a dataframe out of our sorted data
    data = {column_name: feature_ranking_array, "Feature Column Number": column_name_array}
    sorted_feature = pd.DataFrame(data)

    return sorted_feature


def outlier_removal(dataset):
    '''
    This function uses the Z-score to detect and remove the outliers in the
    feature passed into the function
    :param dataset: A dataframe representing the data we will perform
    our function on
    :return: A dataframe that contains the class columns with outliers
    removed.
    '''

    # applying Z-score on our class to find our outliers
    z = np.abs(stats.zscore(dataset))

    # list to append the locations of our outliers
    l = []
    l.append(np.where(z > 3))
    #outlier_removed_row_index = l[0][0]
    #outlier_removed_column_index = l[0][1]

    # the filtered outliers are found using a threshold of 2.5
    filtered_outliers = (z <= 3).all(axis=1)
    removed_outliers_class_df = dataset[filtered_outliers]

    return removed_outliers_class_df, l


def prepare_data(dataset, num_top_features, ranked_features_df):
    """
    This functions prepares the data that gets passed into our Bayes Classifier
    Model.
    :param dataset: A dataframe representing the dataset we are working with
    :param num_top_features: An integer representing the number of top features
    we are working with.
    :param ranked_features_df: A dataframe with the ranked features
    :return: A dataframe to be used for the Bayes Classifier, and a dataframe
    with the top ranked features score along with the column number the
    score belongs to.
    """

    top_features_to_use = retrieve_top_k_features(ranked_features_df,
                                                  num_top_features)["Feature Column Number"].to_list()
    bayes_df_columns = ranked_features_df["Feature Column Number"].to_list()

    # adding a column 0 to account for our label column
    bayes_df_columns.insert(0, "Column_0")

    # creating dataframe to pass into our bayes classifier model
    bayes_df = pd.DataFrame(dataset)
    bayes_df.columns = bayes_df_columns
    return bayes_df, top_features_to_use


def bayes_classifier(dataset, features_cols, target_label):
    """
    This is the Bayes Classifier model that will classify our labels. It uses an
    80/20 split, where 80% of the data is for training, and 20% is for testing.
    :param dataset: A dataframe representing the dataset we are working with
    :param features_cols: A list containing the column name of the top k
    features.
    :param label: An integer or string representing the column number of column
    name of the target label column.
    :return: A float representing the accuracy value of our Bayes Classifier
    model.
    """

    # we check if target label col was passed as column number
    if isinstance(target_label, int):
        label_col = dataset.iloc[:,target_label]
    else:
        label_col = dataset.loc[:,target_label]

    data_cols = dataset.loc[:, features_cols]
    X_train, X_test, y_train, y_test = train_test_split(data_cols,label_col, test_size = 0.2, random_state=109)

    # creating the Gaussian Classifier
    model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy_val =  "{:.2f}".format(metrics.accuracy_score(y_test,y_pred)*100)
    print(f"Our Bayes Classifier model accuracy with top {len(features_cols)} ranked features: {accuracy_val}")
    return accuracy_val

def k_fold_cross_validation_bayes_classifier(dataset, n_folds, target_label_col_number):
    """
    This function runs the Bayes Classifier on different folds of the dataset.
    :param dataset: A dataframe representing the dataset we are working with
    :param n_folds: An integer representing the number of folds
    :param target_label_col_number: An integer representing the target label
    column number.
    :return: A list that contains the accuracy of each fold run
    """

    kf = KFold(n_splits=n_folds, shuffle=False)
    i = 1
    accuracy = []
    for train_index, test_index in kf.split(dataset):
        #print(train_index,test_index)
        X_train = dataset.iloc[train_index,1:61]
        X_test = dataset.iloc[test_index, 1:61]
        y_train = dataset.iloc[train_index, target_label_col_number]
        y_test = dataset.iloc[test_index, target_label_col_number]

        # creating the Gaussian Classifier
        model = GaussianNB()

        model.fit(X_train, y_train)

        accuracy_val =  "{:.2f}".format(metrics.accuracy_score(y_test, model.predict(X_test))*100)
        print(f"Bayes Classifier Model Accuracy for the fold no. {i} on the test set: {accuracy_val}")
        accuracy.append(float(accuracy_val))
        i += 1

    folds = []
    for i in range(n_folds):
        folds.append(i+1)

    # plotting the accuracy on the different fold iterations
    fig = plt.figure()
    plt.plot(folds, accuracy)
    plt.xlabel("Fold Number")
    plt.ylabel("Bayes Model Accuracy")
    plt.title(
        "Model Accuracy vs Fold Iteration our Bayes Classifier Model is Running")
    plt.show()

def support_vector_machine_model(dataset, top_features_to_use, target_label):
    """
    This function runs the Polynomial SVM Classifier on our dataset to classify
    the target label for each of the rows of data we have in our dataset. The
    function uses an 80/20 split on the data, where 80% of the data is for
    training, and 20% is for testing.
    :param dataset: A dataframe representing the dataset we are working with
    :param top_features_to_use:
    :param target_label: An integer or string representing the column number of column
    name of the target label column.
    :return: A float representing the accuracy value of our Bayes Classifier
    model.
    """
    # we check if target label col was passed as column number
    if isinstance(target_label, int):
        label_col = dataset.iloc[:,target_label]
    else:
        label_col = dataset.loc[:,target_label]

    independent_variables = dataset.loc[:,top_features_to_use]
    X_train, X_test, y_train, y_test = train_test_split(independent_variables,
                                                        label_col,
                                                        test_size=0.2,
                                                        random_state=109)

    # Initializing SVM Classifier
    clf = svm.SVC(kernel='poly', degree=10, C=1, decision_function_shape='ovo').fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    clf_model_accuracy = "{:.2f}".format((metrics.accuracy_score(y_test, y_pred))*100)
    print(f"Our SVM Classifier model accuracy with top {len(top_features_to_use)} ranked features: {clf_model_accuracy}")
    return clf_model_accuracy

def k_fold_cross_validation_support_vector_classifier(dataset, n_folds, target_label_col_number):
    """
    This function runs the SVM Classifier on different folds of the dataset.
    :param dataset: A dataframe representing the dataset we are working with
    :param n_folds: An integer representing the number of folds
    :param target_label_col_number: An integer representing the target label
    column number.
    :return: A list that contains the accuracy of each fold run
    """

    kf = KFold(n_splits=n_folds, shuffle=False)
    i = 1
    accuracy = []
    for train_index, test_index in kf.split(dataset):
        #print(train_index,test_index)
        X_train = dataset.iloc[train_index,1:61]
        X_test = dataset.iloc[test_index, 1:61]
        y_train = dataset.iloc[train_index, target_label_col_number]
        y_test = dataset.iloc[test_index, target_label_col_number]

        # Initializing SVM Classifier
        # Initializing SVM Classifier
        clf = svm.SVC(kernel='poly', degree=10, C=1,
                      decision_function_shape='ovo').fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        poly_model_accuracy =  "{:.2f}".format(metrics.accuracy_score(y_test, y_pred)*100)
        print(f"SVM Model Accuracy for the fold no. {i} on the test set: {poly_model_accuracy}")
        accuracy.append(float(poly_model_accuracy))
        i += 1

    folds = []
    for i in range(n_folds):
        folds.append(i+1)

    # plotting the accuracy on the different fold iterations
    fig = plt.figure()
    plt.plot(folds, accuracy)
    plt.xlabel("Fold Number")
    plt.ylabel("SVM Model Accuracy")
    plt.title(
        "Model Accuracy vs Fold Iteration our SVM Model is Running")
    plt.show()


# reading in our dataset
our_dataset = read_dataset("Features42k.csv")

# remove outliers
outliers_removed_df, outliers_removed = outlier_removal(our_dataset)

# retrieving our ranked features dataframe
feature_ranking_df = feature_ranking_chi_squared(outliers_removed_df, False, True)
# displaying the ranking of all features in our dataset
print(feature_ranking_df)

# retrieving top k features, in the case below, top 10 features
top_ten_features = retrieve_top_k_features(feature_ranking_df, 10)
# displaying the ranking of top 10 ranked features
print(top_ten_features)


# structure to store our accuracies for our different runs with our
# Bayes classifier and our SVM model
bayes_classifier_accuracy = []
svm_accuracy = []

top_k_ranked_features_runs = [5,10,20,40,60]
for i in top_k_ranked_features_runs:
    # running bayes classifier and SVM using top 5 ranked features
    df, top_features_to_use = prepare_data(dataset=our_dataset,
                                           num_top_features=i,
                                           ranked_features_df=feature_ranking_df)
    bayes_accuracy_value = bayes_classifier(df, top_features_to_use, target_label=0)
    bayes_classifier_accuracy.append(bayes_accuracy_value)
    svm_accuracy_value = support_vector_machine_model(df, top_features_to_use,
                                                      target_label=0)
    svm_accuracy.append(svm_accuracy_value)


# plotting the accuracy of our bayes classifier and SVM in the different runs
# the different runs have different K used for top K Ranked features
fig = plt.figure()
x_new = np.linspace(5, 60, 200)
spl = make_interp_spline(top_k_ranked_features_runs, bayes_classifier_accuracy, k=2)
y_smooth = spl(x_new)
plt.plot(x_new, y_smooth, label = "Bayes Classifier Model")

spl = make_interp_spline(top_k_ranked_features_runs, svm_accuracy, k=2)
y_smooth = spl(x_new)
plt.plot(x_new, y_smooth, label = "SVM Model")
plt.xlabel("Top K Ranked Features Used")
plt.ylabel("Model Accuracy")
plt.legend(loc=0)
plt.title("Model Accuracy vs Top K Ranked Features Used by our Classifiers")
plt.show()


# Performing 5 fold cross validation and feeding it into our bayes classifier
k_fold_cross_validation_bayes_classifier(our_dataset, 5, 0)

# Performing 5 fold cross validation and feeding it into our SVM classifier
k_fold_cross_validation_support_vector_classifier(our_dataset, 5, 0)
