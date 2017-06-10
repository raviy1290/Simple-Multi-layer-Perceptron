import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

'''
wine data set of 3 diff winery with 13 other properties (csv already downloded)
for properties and data check below URL
https://archive.ics.uci.edu/ml/datasets/Wine
winery martix
'''

wine = pd.read_csv('wine_data.csv', names = ["Winery", "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total phenols", "Falvanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280_315", "Proline"])

#print wine.head()

X = wine.drop("Winery", axis=1)
y = wine["Winery"]

#print X.head()

#use sklearn for spliting data into train and test parts
X_train, X_test, y_train, y_test = train_test_split(X, y)

'''
normalization/scaling of data, since all properties data is in different unit 
StandardScaler or MinMaxScaler or MaxAbsScaler
'''
#standard scaler
scaler = StandardScaler()
#min max scaler
minmax_scaler = MinMaxScaler()
#maxabs scaler
maxabs_scaler = MaxAbsScaler()

# Fit only to the training data set as test data set is dynamic
scaler.fit(X_train)
minmax_scaler.fit(X_train)
maxabs_scaler.fit(X_train)

# Transform all data using required scaler
ss_X_train = scaler.transform(X_train)
ss_X_test = scaler.transform(X_test)

mm_X_train = minmax_scaler.transform(X_train)
mm_X_test = minmax_scaler.transform(X_test)

ma_X_train = maxabs_scaler.transform(X_train)
ma_X_test = maxabs_scaler.transform(X_test)


#Multi layer perceptron classifier NN,
#will use max abs normalized data 
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000)
mlp.fit(ma_X_train,y_train)
predictions = mlp.predict(ma_X_test)

#To describe the performance of a classification model
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))