import streamlit as st 
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df = pd.read_excel("C:\Documents\python data\Bankruptcy.xlsx")
df
df.info()
df.describe()
df.isnull().sum() #checking for null values
df.duplicated()
plt.figure()
sns.countplot(x='class',data=df)
plt.title("Distribution of Bankruptcy status")
plt.show()
L1=LabelEncoder()#initialize the LabelEncoder
df['class_encoded']=L1.fit_transform(df['class'])
# Encode the 'class' variable into numeric form for correlation analysis
labelencoder = LabelEncoder()  # Ensure LabelEncoder is correctly defined and imported
df['class_encoded'] = labelencoder.fit_transform(df['class'])
df2=df.drop_duplicates() #drop the duplicate values in the dataset 
df2
df2.hist(figsize=(8,8))
df2.head()
df2.drop('class', axis=1, inplace=True)
df2
df2 = df2.rename(columns={'class_encoded': 'class'})
corr=df2.corr()
corr
plt.figure(figsize=(8,8))
sns.heatmap(corr,annot=True,linewidth=0.5)
plt.title('correlation Heatmap')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
x=df2[['industrial_risk', 'management_risk','financial_flexibility', 'credibility','competitiveness', 'operating_risk']]
y=df2['class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#initialize the model
model=DecisionTreeClassifier(random_state=42)
#train the model
model.fit(x_train,y_train)
#predict the T.V for testing
y_pred=model.predict(x_test)
y_pred
import subprocess
import sys

# Install streamlit using subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
# Title of the app
st.title('Bankruptcy Prediction App')
# Sidebar for user inputs
st.sidebar.header('Input Company Risk Factors')
# Function to take user inputs
def user_input_features():
    industrial_risk = st.sidebar.selectbox('Industrial Risk', ['Low', 'Medium', 'High'], index=1)
    management_risk = st.sidebar.selectbox('Management Risk', ['Low', 'Medium', 'High'], index=1)
    financial_flexibility = st.sidebar.selectbox('Financial Flexibility', ['Low', 'Medium', 'High'], index=1)
    credibility = st.sidebar.selectbox('Credibility', ['Low', 'Medium', 'High'], index=1)
    competitiveness = st.sidebar.selectbox('Competitiveness', ['Low', 'Medium', 'High'], index=1)
    operating_risk = st.sidebar.selectbox('Operating Risk', ['Low', 'Medium', 'High'], index=1)

    # Convert categorical inputs into numerical values (for model compatibility)
    risk_mapping = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}

    data = {
        'industrial_risk': risk_mapping[industrial_risk],
        'management_risk': risk_mapping[management_risk],
        'financial_flexibility': risk_mapping[financial_flexibility],
        'credibility': risk_mapping[credibility],
        'competitiveness': risk_mapping[competitiveness],
        'operating_risk': risk_mapping[operating_risk]
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input parameters
st.subheader('User Input Parameters')
st.write(input_df)

# For demonstration, training a new model
# Load dataset
df = pd.read_excel('Bankruptcy.xlsx')

# Preprocess data
X = df.drop('class', axis=1)
y = df['class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model 
model = DecisionTreeClassifier()
model.fit(X_scaled, y)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display prediction results
st.subheader('Prediction')
if prediction[0] == 'non-bankruptcy':
    st.write('Non-Bankrupt')
else:
    st.write('Bankrupt')

st.subheader('Prediction Probability')
st.write(prediction_proba)
