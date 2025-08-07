import pandas as pd

'''from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())'''
df4 = pd.read_csv('Datasets\Processed_set.csv') #C:\Users\reziv\Downloads\Lung_Cancer_Model\Lung_Cancer.csv
#df5 = df4.drop(['PatientId'], axis=1)
#df7 = df5.drop(['index'], axis=1)
#df7['Level'] = df7['Level'].replace({'Low': 0, 'Medium': 1, 'High': 2})
#df7.to_csv('C:\\Users\\reziv\\Downloads\\Lung_Cancer_Git\\Scripts\\Processed_Set.csv', index = False)
print(df4.corr()['Level'].sort_values(ascending=False))


X = df4.drop(['Level'], axis=1)
y = df4['Level']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
accuracies = 0
for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)

    mnb = CategoricalNB()
    mnb.fit(X_train, y_train)
    y_prediction = mnb.predict(X_test)
    print(classification_report(y_test, y_prediction))
    print(mnb.score(X_train, y_train))
    accuracy = mnb.score(X_test, y_test)

    #add in parameter
    param_grid = {
        #'var_smoothing': [0.00000001, 0.000000001, 0.00000001, 5e-9],
        'alpha': [1, 1e-1, 1e-2, 1e-3, 1e-4]

        #'fit_prior': [True, False]
    }
    #print(df4)

    grid_search = GridSearchCV(mnb, param_grid, cv=5, scoring = 'accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    alpha = grid_search.best_params_["alpha"]

    tnb = CategoricalNB(alpha=grid_search.best_params_["alpha"])
    tnb.fit(X_train, y_train)
    preds = tnb.predict(X_test)

    #accuracy = metrics.accuracy_score(y_true=y_test, y_pred=preds)
    print("This was the final accuracy:", accuracy)

    #print(grid_search.best_score_)
    accuracies += accuracy

average = accuracies/10
print(average)