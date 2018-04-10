import numpy as np
import re
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


original = pandas.read_csv("train.csv")

original['Embarked'] = original['Embarked'].replace(np.nan, "M", regex=True)
#We fill in missing values with M

original['Age'] = original['Age'].replace(np.nan, original['Age'].mean(), regex=True)
#fill in missing Age with the mean of the column. Perhaps we can do better by predicting the age using other features? 

name = original['Name']
titles = []

for i in range(len(name)):
	s = name[i]
	title = re.search(', (.*)\.', s)
	title = title.group(1)
	titles.append(title)

original['Titles'] = titles
original['Titles'].replace(['Sir', 'Rev', 'Major', 'Lady', 'Jonkheer', 'Dr', 'Don', 'Countess', 'Col', 'Capt'], 'Name')

original['Titles'].replace(['Ms', 'Mme', 'Mlle', 'Master'], ['Miss', 'Mrs', 'Miss', 'Mr'])

Sex_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Sex))
Ticket_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Ticket))
Embarked_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Embarked))
Titles_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Titles))

#Let's try to get a better prediction for age, using the other features of a person:
age_features = pandas.concat([Sex_binarized, original.Pclass, Titles_binarized, original.SibSp, original.Parch], axis=1)
age_features['Age'] = original.Age
age_features = age_features[age_features.Age!=np.nan]

age_result = age_features.Age
age_features = age_features.drop(['Age'], axis=1)

age_x_train, age_x_test, age_y_train, age_y_test = train_test_split(age_features, age_result, test_size = 0.15, random_state = 10)
age_linear_reg = linear_model.LogisticRegression().fit(age_x_train, np.around(age_y_train.values).ravel())
age_prediction = np.subtract(age_y_test, np.around(age_linear_reg.predict(age_x_test)))

#these are the final parameters we will use for our models: 
prediction_params = pandas.concat([Sex_binarized, Ticket_binarized, Embarked_binarized, Titles_binarized, original.Pclass, original.Age, original.SibSp, original.Fare, original.Parch], axis=1)
prediction_result = original.Survived

x_train, x_test, y_train, y_test = train_test_split(prediction_params, prediction_result, test_size = 0.15, random_state = 10)

#simple logistic regression
logistic_model = linear_model.LogisticRegression().fit(x_train, y_train.values.ravel())
logistic_prediction = logistic_model.predict(x_test)

#MLPClassifier 
clf = MLPClassifier(activation = 'relu', solver='lbfgs', hidden_layer_sizes=(150), random_state=10)
#our dataset is relatively small. We could consider using a stochastic gradient descent for larger datasets
clf.fit(x_train, y_train)
neural_prediction = clf.predict(x_test)

#Random Forest Sampling
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)

#K Nearest Neighbours 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
k_neigh = knn.predict(x_test)

#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_prediction = dt.predict(x_test)

