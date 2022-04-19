import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
data_df = train_df.append(test_df)
data_df.columns
data_df['Title'] = data_df['Name']
#create a new feature called Title, it helps with the age imputation
#Extract the title from the name after a little clean up
data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True) # the regular expression relies on the period (.) at the end of the title to extract the title
data_df['Title'].unique()

mapping = {'Dr':'Mr','Rev':'Mr','Mlle':'Miss','Major':'Mr','Col':'Mr','Sir':'Mr','Don':'Mr','Mme':'Miss','Jonkheer':'Mr','Lady':'Mrs','Capt':'Mr','Countess':'Mrs','Ms':'Miss','Dona':'Mrs'}
data_df.replace({'Title':mapping},inplace=True)

titles = ['Master', 'Miss', 'Mr', 'Mrs'] 
#titles = data_df['Title'].unique()
age_impute = data_df.groupby('Title')['Age'].median()


for title in titles:
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_impute[titles.index(title)]

#update training data with imputed age values
len(train_df)
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

data_df.drop('Title',axis = 1, inplace = True)
#Family size is a combination of parents and siblings Parch + SibSp
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]

#group people by their last name
data_df['Last_Name'] = data_df['Name'].apply(lambda x:str.split(x,",")[0])


DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
# grp is identifier (Last_name, Fare paid), grp_df is the survived, passid , parch, sibsp, age of that group (the family)
for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId','Parch','SibSp', 'Age', 'Cabin']].groupby(['Last_Name','Fare']):
    if (len(grp_df) != 1): #more than one in a group
       for ind, row in grp_df.iterrows():
           smax = grp_df.drop(ind)['Survived'].max()
           smin = grp_df.drop(ind)['Survived'].min()
           passID = row['PassengerId']
           if (smax == 1.0):
               data_df.loc[data_df['PassengerId'] == passID,'Family_Survival'] = 1
           elif (smin == 0.0):
               data_df.loc[data_df['PassengerId']== passID, 'Family_Survival'] = 0
print("Number of passengers with family survival data:", data_df.loc[data_df['Family_Survival'] != 0.5].shape[0])


ballsack = data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId','Parch','SibSp', 'Age', 'Cabin']].groupby(['Last_Name','Fare'])

data_df.loc[data_df['Last_Name']=='Abbing']

#Group by ticket as well to set Family Survival
for balls, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID,'Family_Survival'] = 1
                elif(smin == 0.0):
                    data_df.loc[data_df['PassengerId'] == passID,'Family_Survival'] = 0
print("Number of passengers with family/group survival information: " + str(data_df[data_df['Family_Survival'] != 0.5].shape[0]))                               

#Add this family survival data to the train/test data sets
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]

