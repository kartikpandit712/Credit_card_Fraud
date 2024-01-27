import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

'''
Dataset Download link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3
'''

#Loading dataset
def load_dataset():
    dataset = pd.read_csv('creditcard data.csv')
    return dataset

#Shape of dataset
def dataframe_shape():
    return load_dataset().shape

#Check null values
def sum_of_null_values():
    return load_dataset().isnull().sum()

#Check datatypes
def check_datatypes():
    return load_dataset().dtypes

#Describe data
def data_describe():
    return load_dataset().describe()

#Check count of target variable
def check_count_of_target_variable():
    return load_dataset()['Class'].value_counts()

#Correlation matrix
def corr_matrix():
    return load_dataset().corr()

#Plot target count
def plot_target_count():
    return sns.countplot(data=load_dataset(), x='Class')

#Feature scaling
def feature_scaling_amount():
    data = load_dataset()
    sc = StandardScaler()
    scaled = sc.fit_transform(data['Amount'].values.reshape(-1,1))
    data['Amount'] = scaled
    return data

#Drop unnecessary columns
def drop_unnecessary_columns():
    data = load_dataset()
    data.drop('Time',axis=1,inplace=True)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
    return data

#Drop duplicates
def drop_duplicate_data():
    data = drop_unnecessary_columns()
    data = data.drop_duplicates()
    return data

#Feature separating
def feature_separating_x_y():
    data = drop_duplicate_data()
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return X,y

#Data balancing
def data_balancing_smote():
    X,y = feature_separating_x_y()
    X_res, y_res = SMOTE().fit_resample(X,y)
    return X_res, y_res

#Splitting dataset
def splitting_dataset():
    X_res, y_res = data_balancing_smote()
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 4)
    return X_train, X_test, y_train, y_test

#Logistic Regression
def fit_logistic_regression():
    X_train, X_test, y_train, y_test = splitting_dataset()
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred)

#Linear Discriminant Analysis
def fit_lda():
    X_train, X_test, y_train, y_test = splitting_dataset()
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred)

#KNN
def fit_knn():
    X_train, X_test, y_train, y_test = splitting_dataset()
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred)

#Decision Tree
def fit_decision_tree():
    X_train, X_test, y_train, y_test = splitting_dataset()
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred)

#Gaussian Naive Bayes
def fit_GaussianNB():
    X_train, X_test, y_train, y_test = splitting_dataset()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred)

#Random Forest
def fit_random_forest():
    X_train, X_test, y_train, y_test = splitting_dataset()
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred)

#Calling all Classification algorithms
print('Scores of Logistic Regression is',fit_logistic_regression())
print('Scores of Linear Discriminant Analysis is',fit_lda())
print('Scores of K Nearest Neighbor is',fit_knn())
print('Scores of Decision Tree is',fit_decision_tree())
print('Scores of Gaussian Naive Bayes is',fit_GaussianNB())
print('Scores of Random Forest Classifier is',fit_random_forest())

#Best models for this datset is Logistic, LDA and Gaussian NB. KNN, Decision Tree and Random Forest are overfitting.