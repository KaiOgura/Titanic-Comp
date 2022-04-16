# Titanic Competition
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')
testfile = pd.read_csv('test.csv')
df.head(10)
df.tail(10)

#Check for null values
df.isnull()['Age']

#Look at Data - Data Visualisation
sns.relplot(x = 'Pclass', y='Survived', data=df)
plt.show()

#Change Age null values to mean of Age
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'] = df['Age'].astype('int32')

#Get dummy variable for sex and Pclass
colms = list(df.columns)
df['female'] = pd.get_dummies(df['Sex'])['female']
df = pd.get_dummies(df, columns = ['Pclass'])
df=df.drop(columns = 'Pclass_3')

#Define our Training and test data sets
train = df.filter(['Age','female', 'Pclass_1', 'Pclass_2'],axis=1)
test = df['Survived']

#Using a logit Model
import statsmodels.api as sm
y = test
X = train
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#SKLearn
#Split the data between train and test variables
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# Create the classification model
regr = LogisticRegression()
#Fit the model with the data - Actually get the coefficients
regr.fit(X_train,y_train)
#What are the coefficients on the variables?
regr.coef_

#Evaluate the Model - 
    # Matrix of probabilities that the predicted output is zero , one for the test set
print(regr.predict_proba(X_test))
#Actual prediction of the model
pred = regr.predict(X_test)
#Score the model - The accuracy of the model
    #takes the input and output as arguments and returns the ratio of the number of correct predictions to the number of observations.
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(regr.score(X_test, y_test)))
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(regr.score(X_train, y_train)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
print(cm)

# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(cm)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# plt.show()

 #For a more comprehensive report on the classification:
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


####MAKING An Array 2D
# x = np.arange(10).reshape(-1, 1)

#Now do it on the test file
testfile = pd.read_csv('test.csv')
testfile['Age'] = testfile['Age'].fillna(testfile['Age'].mean())
testfile['female'] = pd.get_dummies(testfile['Sex'])['female']
testfile = pd.get_dummies(testfile, columns = ['Pclass'])
testfile=testfile.drop(columns = 'Pclass_3')
testdata = testfile.filter(['Age','female','Pclass_1','Pclass_2'])
regr.predict(testdata)
testfile['Survive Pred'] = regr.predict(testdata)

#Did any males get a survival prediction?
testfile.loc[(testfile['Sex']=='male')&(testfile['Survive Pred']==1)]
#3rd class survivers
testfile.loc[(testfile['Pclass_2']==0)& (testfile['Pclass_1']==0)&(testfile['Survive Pred']==1)]


#References
#https://realpython.com/train-test-split-python-data/
#https://realpython.com/logistic-regression-python/
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#https://pbpython.com/categorical-encoding.html
