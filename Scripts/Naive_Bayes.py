import pandas as pd

'''from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())'''
df4 = pd.read_csv('C:\\Users\\reziv\\Downloads\\Lung_Cancer_Git\\Datasets\\Processed_Set.csv') #C:\Users\reziv\Downloads\Lung_Cancer_Model\Lung_Cancer.csv
#df5 = df4.drop(['PatientId'], axis=1)
#df7 = df5.drop(['index'], axis=1)
#df7['Level'] = df7['Level'].replace({'Low': 0, 'Medium': 1, 'High': 2})
#df7.to_csv('C:\\Users\\reziv\\Downloads\\Lung_Cancer_Git\\Scripts\\Processed_Set.csv', index = False)
print(df4.corr()['Level'].sort_values(ascending=False))

X = df4.drop(['Level'], axis=1)
y = df4['Level']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2, shuffle=True)
from sklearn.naive_bayes import GaussianNB
mnb = GaussianNB()
mnb.fit(X_train, y_train)
y_prediction = mnb.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_prediction))

print(mnb.score(X_train, y_train))
print(mnb.score(X_test, y_test))

#add in parameter
param_grid = {
    'var_smoothing': [0.00000001, 0.000000001, 0.00000001, 5e-9],
    #'alpha': list(range(1, 201)),
    #'fit_prior': [True, False]
}
from sklearn.model_selection import GridSearchCV
#print(df4)

grid_search = GridSearchCV(mnb, param_grid, cv=5, scoring = 'accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
