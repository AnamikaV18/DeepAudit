#Credit Card Transaction Fraud Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('C:/Users/anami/OneDrive/Desktop/ML/Credit Card Fraud.csv')
print(df.head())

print(df.describe())
print(df.isnull().sum())

#Countplot about number of fraudulent transactions
sns.countplot(x='Fraud',data = df)
plt.title('Distribution of Fraudulent Transactions')
plt.show()

#Distribution of Transaction Amount by Fraud Status
sns.histplot(data =df, x='TransactionAmount', hue = 'Fraud', multiple='stack',bins=30)
plt.title('Transaction Amount Distribution by Fraud Status')
plt.show()

#Box Plot for transaction amount by fraud status
sns.boxplot(x='Fraud',y='TransactionAmount',data=df)
plt.title(' Box Plot of Transaction Amount By Fraud Status')
plt.show()

#Correlation Matrix
num_cols = list()

for column in df.columns:
    if df[column].dtype != object :
        num_cols.append(column)

correlation_matrix = df[num_cols].corr()
sns.heatmap(correlation_matrix, annot = True, cmap ='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Data PreProcessing
#Encode Categorical variables
label_encoders = {}
for column in ['MerchantCategory','CustomerGender','Location'] :
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

#Featurs and Target
x = df.drop(['TransactionID','Fraud','CustomerGender'],axis=1)
y = df['Fraud']

print(x.head())

#Split training and testing data
x_train , x_test, y_train ,  y_test =train_test_split(x,y,test_size=0.3,random_state=0)

#Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Logistic Regression 

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
logreg_pred = logreg.predict(x_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)

#Display Accuaracy and Performance metrics
print(f'Logistic Regression Accuracy : {logreg_accuracy*100:.2f}')
print('\nLogistic Regression Clasification Report')
print(classification_report(y_test, logreg_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, logreg_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap='Purples')  # You can change the colormap
plt.title("Confusion Matrix")
plt.show()

#Testing the model
TransactionAmount = float(input("Transaction Amount : "))
TransactionTime = float(input("Transaction Time : "))
MerchantCategory = input("Merchant Category : ")
CustomerAge = int(input("Customer Age : "))
AnnualIncome = int(input("Annual Income : "))
Location = input("Location : ")
PreviousFraudCount = int(input("Previous Fraud Count : "))

merchant = label_encoders['MerchantCategory'].transform([MerchantCategory])[0]
location = label_encoders['Location'].transform([Location])[0]

custom = np.array([[TransactionAmount , TransactionTime, merchant, CustomerAge, AnnualIncome, location, PreviousFraudCount]])
custom1 = scaler.transform(custom)
prediction = logreg.predict(custom1)
probability = logreg.predict_proba(custom1)

print("The Credit Card Transaction you've made is most likely ")
if prediction[0] == 1 :
    print("FRAUDULENT")
else :
    print("NOT FRAUDULENT")

print(f"Fraud Probability: {probability[0][1]*100:.2f}%")