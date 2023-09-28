import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #used for we need to convert the mail data into numerical numbers so the ML model understands 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the csv file from pandas data frame
raw_mail_data = pd.read_csv('mail_data.csv')

#relplace the null string vlues with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

#printing the first 5 rows of the data set
mail_data.head()

#checking the number the rows and columns in the dataset 
mail_data.shape #shape gives u the number of rows and colunm in the dataframe
#output:(5572, 2)

# label spam mail ---> 0 and ham mail ---> 1
mail_data.loc[mail_data['Category'] == 'spam','category'] = 0
mail_data.loc[mail_data['Category'] == 'ham','category'] = 1

# seperating the message and labels i.e seprating the data as text and labels
X = mail_data['Message'] # inputfunction 
Y = mail_data['Category'] # target function ie predection output


#splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
#print(X.shape)           output:(5572,)
#print(X_train.shape)     output:(4457,)
#print(X_test.shape)      output:(1115,)


#feature extraction --> converting the message into meaningful numerical values
#transform the text data feature vector that can be used as input for the logisticRegression model
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integer , its is already in 0's and 1's... becoz some tym it may consier has string 

# Y_train = Y_train.astype('int')
# Y_test = Y_test.astype('int')



#Traing the logistic regression machinelearning model
model = LogisticRegression()
#training the logistic regression model with the training data
model.fit(X_train_features, Y_train)



#predicting the training data
predection_on_training_data = model.predict(X_train_features)

#compare the predected value by actual value
accuracy_on_training_data = accuracy_score(Y_train,predection_on_training_data)
#print("accuracy on training data : ",accuracy_on_training_data)
#output: accuracy on training data :  0.9670181736594121 --->96% which is very gud accuracy score


#predicting the test data
predection_on_test_data = model.predict(X_test_features)

#compare the predected value by actual value
accuracy_on_test_data = accuracy_score(Y_test,predection_on_test_data)
#print("accuracy on test data : ",accuracy_on_test_data)
#output: accuracy on test data :  0.9659192825112107 -->96%

#building a predictive system 
# if  we give a new mail it will predict wether its is spam or not 
input_mail = ["URGENT! We are trying to contact you. Last weekends draw shows that you have won a £900 prize GUARANTEED. Call 09061701939. Claim code S89. Valid 12hrs only"]

# convert text to features vector
input_data_features = feature_extraction.transform(input_mail)

#making prediction
prediction = model.predict(input_data_features)
print(prediction)


if prediction[0]=="ham":
  print("The mail is 'HAM MAIL'")
else:
  print("The mail is 'SPAM MAIL'")