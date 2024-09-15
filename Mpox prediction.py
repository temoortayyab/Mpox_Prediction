
# data analysis libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.interactive(False)

# Preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Machine Learning Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# metric evaluation libraries
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve, RocCurveDisplay

## Set Working directory
import os
os.chdir("C:\\Users\\temoo\\OneDrive\\Desktop\\Uni\\Year MPH 1\\Mpox prediction")
data = pd.read_csv('DATA.csv')

import warnings
warnings.filterwarnings('ignore')

# shape
print(data.shape)
# head
print(data.head(20))
# descriptions
print(data.describe())
# class distribution
print(data.groupby('MonkeyPox').size())


print(data.isnull().sum()) #None for systemic illness is being interpreted as missing, no missing data otherwise

#Change "None" to "No"
data['Systemic Illness'] = data['Systemic Illness'].fillna('No')

print(data.isnull().sum()) #None for systemic illness is being interpreted as missing, no missing data otherwise
#No missing data now

#removing patient_ID
data = data.drop(columns=["Patient_ID"], axis=1)

#one-hot encoding for systematic illness variable
data = pd.get_dummies(data,columns=['Systemic Illness'])
print(data.head())

#explanding display
pd.set_option('display.max_columns', None)

#recoding T/F and Positive/Negative to 1s and 0s
data = data.replace(["Positive", "Negative", True, False], [1,0,1,0])

#removing the Systematic Illness_No variable as it would be redundant with the other categories, causing multicollinearity
data = data.drop(columns=["Systemic Illness_No"], axis=1)

#Exploratory Data Analysis

#Class balance
palette = sns.color_palette('pastel')
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(y='MonkeyPox', data=data, palette=palette, ax=ax)
ax.set_title('Distribution of Monkey Pox', fontsize=15);

#Univariate plots
plt.figure(figsize=(12, 8))

for i, column in enumerate(data.columns, 1):
    plt.subplot(len(data.columns), 1, i)
    plt.hist(data[column], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)

plt.tight_layout()
plt.show(block=True)

#correlation matrix
from matplotlib.pyplot import figure
plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), cmap=plt.cm.CMRmap_r, annot = True)
plt.show(block=True)
#No features/variables are significantly correlated with each other

#Splitting Data into Training and Testing
X_train=data.drop(columns=["MonkeyPox"])
y_train=data["MonkeyPox"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
print('Train dataset shape:',X_train.shape)
print('Test dataset shape', y_train.shape)

#Identifying numeric and categorical columns
numeric_columns = X_train.select_dtypes(exclude='object').columns
print(numeric_columns)
print('*'*100)
categorical_columns = X_train.select_dtypes(include='object').columns
print(categorical_columns)

numeric_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='median')),
    ('scaling',StandardScaler(with_mean=True))
])

print(numeric_features)
print('*'*100)

categorical_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='most_frequent')),
    ('encoding', OneHotEncoder()),
    ('scaling', StandardScaler(with_mean=False))
])

print(categorical_features)

processing = ColumnTransformer([
    ('numeric', numeric_features, numeric_columns),
    ('categorical', categorical_features, categorical_columns)
])

processing

#Model Preparation and Evaluation
def prepare_model(algorithm):
    model = Pipeline(steps=[
        ('processing', processing),
        ('pca', TruncatedSVD(n_components=3, random_state=12)),
        ('modeling', algorithm)
    ])
    model.fit(X_train, y_train)
    return model


def prepare_confusion_matrix(algo, model):
    print(algo)
    plt.figure(figsize=(12, 8))
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    plt.show(block=True)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');

def prepare_classification_report(algo, model):
    print(algo+' Report :')
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))
def prepare_roc_curve(algo, model):
    print(algo)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    roc_auc = auc(fpr, tpr)
    curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    curve.plot()
    plt.show(block=True)

#Model Training
algorithms = [('bagging classifier', BaggingClassifier()),
              ('KNN classifier', KNeighborsClassifier()),
              ('Random Forest calssifier', RandomForestClassifier()),
              ('Adaboost classifier', AdaBoostClassifier()),
              ('Gradientboot classifier',GradientBoostingClassifier()),
              ('MLP', MLPClassifier()),
              ('DecisionTree classifier', DecisionTreeClassifier())
             ]

trained_models = []
model_and_score = {}

for index, tup in enumerate(algorithms):
    model = prepare_model(tup[1])
    model_and_score[tup[0]] = str(model.score(X_train,y_train)*100)+"%"
    trained_models.append((tup[0],model))

#Model Evaluation
for key in model_and_score:
    print(key+" = "+model_and_score[key])

for index, tup in enumerate(trained_models):
    prepare_confusion_matrix(tup[0], tup[1])