# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, linear_model
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, accuracy_score, roc_curve
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn import model_selection
from sklearn import tree
import statsmodels.api as sm
import os
from sklearn.metrics import f1_score, make_scorer, recall_score, fbeta_score
from sklearn.tree.export import export_text

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#%%
print(os.getcwd())
os.chdir("Downloads/HD_DF/")
#os.chdir('HD_DF')
#%%

df = pd.read_excel('HeartDiseasePrediction.xlsx')
print(df.columns)
print(df.head(n=10))

#Delete rows with missing target values
df = df[df['TenYearCHD'].notna()]

#Impute NA
print(df.isna().sum())

#Male column has 1,0,M, F, and NA - this needs to be cleaned up
mf_dict = {'M': 1, 'F': 0}
df['male'] = df.replace({"male": mf_dict})
df['male'] = df.fillna(df['male'].mode())

#BPMeds
df['BPMeds'] = df.fillna(df['BPMeds'].mode())
#Education
df['education'] = df.fillna(df['education'].mode())

#One column (BP) is nearly 50% NAs I think this should be eliminated
df.drop("BP", 1, inplace = True)
df_clean = df.fillna(df.mean())

Y = df_clean['TenYearCHD']
X = df_clean.drop('TenYearCHD', axis = 1)
print(X.columns)

train_x, test_x, train_y, test_y = train_test_split(X, Y)

#%%
print(df.columns)
#%%
#We record absence or presence of a dominant 
#genetic marker in binary form 
#and instead of calculating correlation, 
#we calculate resemblance (similarity) 
#coefficients (simple matching (SM), Dice, Jaccard etc.).
print(df_clean.columns)
#df_clean.hist(grid = False)
#plt.show()

#Label data that is continuous and binary
binary_data = [i for i in df_clean[df_clean.columns] if len(df[i].unique()) == 2]
cont_data = [i for i in df_clean[df_clean.columns] if len(df[i].unique()) != 2]
print(cont_data)
print(binary_data)
#%% Exploratory Data Analysis - Histograms
df_clean[cont_data].hist(grid=False)
plt.show()

#%% Exploratory Data Analysis - Continuous Variables 
fig, axs = plt.subplots(2,2)
fig.suptitle('Exploratory Data Analysis')
sns.distplot(df['age'], ax=axs[0,0])
sns.distplot(df['heartRate'], ax=axs[1,0])
sns.distplot(df['totChol'], ax=axs[0,1])
sns.distplot(df['cigsPerDay'], ax=axs[1,1])
plt.show()

#%% Summarize by age for some exploratory data analysis
summarize_df = df_clean.groupby('age', as_index=False).agg({'totChol': 'mean',
                                                            'heartRate': 'mean',
                                                            'BMI': 'mean',
                                                            'glucose': 'mean',
                                                            'TenYearCHD': 'mean'})

fig, axs = plt.subplots(4,1)
plot1 = sns.regplot(x='age', y='totChol', data=summarize_df, ax=axs[0], 
                    color = 'darkgreen')
plot1.set_ylabel('Total \n Cholesterol')
plot2 = sns.regplot(x='age', y='BMI', data=summarize_df, ax=axs[1],
                    color = 'darkblue')
plot3 = sns.regplot(x="age",y='glucose', data=summarize_df, ax=axs[2],
                    color = 'darkred')
plot3.set_ylabel('Glucose')
plot4 = sns.regplot(x="age",y='TenYearCHD', data=summarize_df, ax=axs[3],
                    color = 'black')
plot4.set_ylabel('Ten Year \n CHD Prob')
plt.tight_layout()
#plt.show()

#%% Scorers
recall_met = make_scorer(recall_score, average='micro') #recall should minimize false neg
ftwo_scorer = make_scorer(fbeta_score, beta=2, average='micro')
f1 = make_scorer(f1_score , average='micro')

#%%
#Compare to no information gained with our data
count_CHD = len([i for i in Y if i == 1])
count_no_CHD = len([j for j in Y if j == 0])
naive_pos_rate = count_CHD/(count_CHD+count_no_CHD)
print(naive_pos_rate)

#%% Run DT Classifier with oversampled DataFrame
#Naive Random Over-Sampling to run against regular dataset

print(len(df_clean[df_clean['TenYearCHD'] == 1]))
print(len(df_clean[df_clean['TenYearCHD'] == 0]))

print(df_clean[df_clean['TenYearCHD'] == 1])

#644/4300 pieces of data are 1
#Let's oversample these!

oversampled_df = df_clean.append(df_clean[df_clean['TenYearCHD'] == 1], ignore_index=True)
#We have now counted each minority target twice!

Y_oversampled = oversampled_df['TenYearCHD']
X_oversampled = oversampled_df.drop('TenYearCHD', axis = 1)

oversampled_df2 = oversampled_df.append(df_clean[df_clean['TenYearCHD'] == 1], ignore_index=True)
#We have now counted each minority target twice!

Y_oversampled2 = oversampled_df2['TenYearCHD']
X_oversampled2 = oversampled_df2.drop('TenYearCHD', axis = 1)
#%%
param_dist = {'max_depth': [1,2,3,4,5,6,7,8,9,10],
              "min_samples_leaf": np.arange(1,9),
              "criterion": ['gini','entropy']}

clf = GridSearchCV(DecisionTreeClassifier(), 
                   param_dist, scoring = ftwo_scorer,
                   cv = 10, return_train_score = False)

#{'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 6}

#%% Make function to summarize DT Performance
def summarize_DT_performance(scorer, x_data, y_data):
    clf = DecisionTreeClassifier(max_depth = 3, 
                                min_samples_leaf = 6,
                                criterion = 'entropy',
                                random_state = 0)
    
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data)
    
    model = clf.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    sensitivity = tp/(tp+fn) #true positive rate
    specificity = tn/(tn+fp) #true negative rate

    print('sensitivity:', round(sensitivity,4),
          'specificity:', round(specificity,4))
    
#%%
    
summarize_DT_performance(f1, X, Y)
summarize_DT_performance('roc_auc', X, Y)
summarize_DT_performance(ftwo_scorer, X, Y)

summarize_DT_performance(f1, X_oversampled, Y_oversampled)
summarize_DT_performance('roc_auc', X_oversampled, Y_oversampled)
summarize_DT_performance(ftwo_scorer, X_oversampled, Y_oversampled)

summarize_DT_performance(ftwo_scorer, X_oversampled2, Y_oversampled2)
summarize_DT_performance(ftwo_scorer, X_oversampled2, Y_oversampled2)
summarize_DT_performance(ftwo_scorer, X_oversampled2, Y_oversampled2)

#%% Find out what the best decision tree model was split on
# Visualize the decision tree
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6,
                            criterion='entropy')
clf = dt.fit(train_x, train_y)
tree.plot_tree(clf,
               feature_names=list(train_x),
               filled=True); #This model just gives columns by index so let's find out what the splits are on by name

tree_rules = export_text(clf, feature_names=list(train_x))

#%% RandomForest find best parameters
rf = RandomForestClassifier(random_state=0)
param_dist = {'max_depth' : [5, 10, 20],
              'max_features' : ['auto', 5, None],
              'n_estimators' :[5, 10, 15, 20]}

clf = GridSearchCV(RandomForestClassifier(), 
                    param_dist, 
                   cv = 10, return_train_score = False, 
                   scoring = f1) 
clf.fit(X,Y)
print(clf.best_params_)

#%% Functionalize ML Performance
f1 = make_scorer(f1_score , average='macro')

rf = RandomForestClassifier(max_depth = 20, 
                                max_features = 5,
                                n_estimators = 20)

def summarize_RF_performance(scorer, x_data, y_data):
    clf = RandomForestClassifier(max_depth = 20, 
                                max_features = 5,
                                n_estimators = 20,
                                random_state = 0)
    
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data)

    
    model = clf.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    sensitivity = tp/(tp+fn) #true positive rate
    specificity = tn/(tn+fp) #true negative rate

    print('metric:', scorer,
          'sensitivity:', round(sensitivity,4),
          'specificity:', round(specificity,4),
          '\n')
#%%
summarize_RF_performance(f1, X, Y)
summarize_RF_performance('roc_auc', X, Y)
summarize_RF_performance(ftwo_scorer, X, Y)

summarize_RF_performance(f1, X_oversampled, Y_oversampled)
summarize_RF_performance('roc_auc', X_oversampled, Y_oversampled)
summarize_RF_performance(ftwo_scorer, X_oversampled, Y_oversampled)

summarize_RF_performance(f1, X_oversampled2, Y_oversampled2)
summarize_RF_performance('roc_auc', X_oversampled2, Y_oversampled2)
summarize_RF_performance(ftwo_scorer, X_oversampled2, Y_oversampled2)
