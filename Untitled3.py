#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
df = pd.read_excel(r'C:/Users/Lenovo/Downloads/Datasetmedi.xlsx')
df.head()


# In[3]:


# Display basic information about the dataset
df.info()


# In[4]:


# Check for any missing values in the dataset
df.isnull().sum()


# In[5]:


# Get summary statistics for numerical columns
df.describe()


# In[6]:


# Check the data types of each column
df.dtypes


# In[7]:


# Get the number of unique values in each column to identify categorical features
df.nunique()


# In[8]:


# Example: For categorical columns, you might want to fill missing values with the mode
df['contact'].fillna(df['contact'].mode()[0], inplace=True)

df['duration'].fillna(df['duration'].median(), inplace=True)
df['poutcome'].fillna(df['poutcome'].mode()[0], inplace=True)
df.dropna(subset=['Term Deposit'], inplace=True)
df['Count_Txn'].fillna(df['Count_Txn'].median(), inplace=True)
df['job'].fillna(df['job'].mode()[0], inplace=True)
df['marital'].fillna(df['marital'].mode()[0], inplace=True)
df['education'].fillna(df['education'].mode()[0], inplace=True)

# Convert 'balance' to numeric, coercing errors to NaN
df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
# Calculate the median of the non-NaN values
median_balance = df['balance'].median()
# Fill NaN values with the median
df['balance'].fillna(median_balance, inplace=True)

# Convert 'Annual Income' to numeric, coercing errors to NaN
df['Annual Income'] = pd.to_numeric(df['Annual Income'], errors='coerce')
# Calculate the median of the non-NaN values
median_income = df['Annual Income'].median()
# Fill NaN values with the median
df['Annual Income'].fillna(median_income, inplace=True)


# In[9]:


# Alternatively, drop specific columns which are not required for analysis
df.drop(columns=['Sno'], inplace=True)


# In[10]:


print(df.isnull().sum())


# In[11]:


# Removing the duplicates values from the dataset
# Standardize job types
df['job'] = df['job'].replace({'blue collar': 'blue-collar'})
# Standardize education types
df['education'] = df['education'].replace({'ter tiary': 'tertiary'})
# Standardize contact types
df['contact'] = df['contact'].replace({'Tel': 'telephone'})
# Replace question marks with NaN
df['poutcome'].replace('?', np.nan, inplace=True)
df['poutcome'].replace('????', np.nan, inplace=True)
# Display DataFrame after replacing question marks
print("\nDataFrame after replacing question marks with NaN:")
print(df['poutcome'].value_counts(dropna=False))
# Fill NaN values with the mode (most frequent value)
mode_value = df['poutcome'].mode()[0]
df['poutcome'].fillna(mode_value, inplace=True)


# In[12]:


#  Fill NaN values with the mode (most frequent value)
mode_value = df['contact'].mode()[0]
df['contact'].fillna(mode_value, inplace=True)
print(df['contact'].value_counts(dropna=False))
# Standardize contact types
df['contact'] = df['contact'].replace({'Tel': 'telephone'})
# Check the unique values in the 'contact','job','education' and 'term_deposit' columns
print(df['contact'].unique())
print(df['Term Deposit'].unique())
print(df['job'].unique())
print(df['education'].unique())


# In[13]:


# Filter customers with no annual income
no_income_customers = df[df['Annual Income'] == 0]
# Count the number of customers with no annual income
num_no_income_customers = no_income_customers.shape[0]
print(f"Number of customers with no annual income: {num_no_income_customers}")


# In[14]:


import matplotlib.pyplot as plt
# Assuming your dataset is loaded in a DataFrame called `df`
# Replace 'Annual Income' with the actual column name
no_income_df = df[df['Annual Income'] == 0]  # Filter customers with no income
# Plot the age distribution of these customers
plt.figure(figsize=(10, 6))
plt.hist(no_income_df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Customers with No Annual Income')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()


# In[15]:


# Filter customers who don't have any type of loan
loan_less_customers = df[df['loan'] == 'no']
# Count the number of loan-less customers
num_loan_less_customers = loan_less_customers.shape[0]
print(f"Number of customers without any loan: {num_loan_less_customers}")


# In[16]:


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
# Plot the income distribution of loan-less customers
plt.figure(figsize=(10, 6))
sns.histplot(loan_less_customers['Annual Income'], kde=True, bins=30, color='green')
plt.title('Income Distribution of Loan-less Customers')
plt.xlabel('Annual Income')
plt.ylabel('Number of Customers')
plt.show()


# In[17]:


# Plot the balance distribution of loan-less customers
plt.figure(figsize=(10, 6))
sns.histplot(loan_less_customers['balance'], kde=True, bins=30, color='blue')
plt.title('Balance Distribution of Loan-less Customers')
plt.xlabel('Balance')
plt.ylabel('Number of Customers')
plt.show()


# In[18]:


# Plot the profession distribution of loan-less customers
plt.figure(figsize=(10, 6))
sns.countplot(y=loan_less_customers['job'], order=loan_less_customers['job'].value_counts().index, palette='viridis')
plt.title('Profession Distribution of Loan-less Customers')
plt.xlabel('Number of Customers')
plt.ylabel('Profession')
plt.show()


# In[19]:


# Filter customers with a loan
loan_customers = df[df['loan'] == 'yes']

# Calculate the percentage of loan customers who have insurance
insurance_customers = loan_customers[loan_customers['Insurance'] == 'yes']
percentage_with_insurance = (insurance_customers.shape[0] / loan_customers.shape[0]) * 100

print(f"Percentage of customers with a loan who have taken out insurance: {percentage_with_insurance:.2f}%")


# In[20]:


import matplotlib.pyplot as plt
# Calculate the number of customers with and without insurance
insurance_counts = loan_customers['Insurance'].value_counts()
# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(insurance_counts, labels=insurance_counts.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Insurance Status of Customers with a Loan')
plt.show()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Group by contact method and calculate the success rate for each method
contact_success_rate = df.groupby('contact')['Term Deposit'].value_counts(normalize=True).unstack().fillna(0)
# Calculate the success percentage (Term Deposit = 'yes')
contact_success_rate['success_percentage'] = contact_success_rate['yes'] * 100
# Sort by success percentage in descending order
contact_success_rate = contact_success_rate.sort_values(by='success_percentage', ascending=False)
# Print the success rates
print(contact_success_rate[['success_percentage']])


# In[22]:


# Filter out rows with "?" and "Mobile" from the data
filtered_contact_success_rate = contact_success_rate[~contact_success_rate.index.isin(['?'])]
# Plot the success percentage for each contact method
plt.figure(figsize=(10, 6))
sns.barplot(x=filtered_contact_success_rate.index, y=filtered_contact_success_rate['success_percentage'], palette='viridis')
plt.title('Success Rate of Each Contact Method for Term Deposit Subscriptions')
plt.xlabel('Contact Method')
plt.ylabel('Success Percentage (%)')
plt.xticks(rotation=45)
plt.show()


# In[23]:


# Define age bins and labels
bins = [18, 30, 40, 50, 60, 70, 80]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79']
# Create a new column for age groups
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
# Display the new column
df[['age', 'age_group']].head()


# In[24]:


# Calculate the percentage of customers with a home loan in each age group
age_group_home_loan = df.groupby('age_group')['housing'].value_counts(normalize=True).unstack().fillna(0)

# Calculate the percentage of home loans
age_group_home_loan['home_loan_percentage'] = age_group_home_loan['yes'] * 100

# Sort by home loan percentage in descending order
age_group_home_loan = age_group_home_loan.sort_values(by='home_loan_percentage', ascending=False)

# Print the percentage of home loans by age group
print(age_group_home_loan[['home_loan_percentage']])


# In[25]:


# Plot the percentage of home loans by age group
plt.figure(figsize=(10, 6))
sns.barplot(x=age_group_home_loan.index, y=age_group_home_loan['home_loan_percentage'], palette='coolwarm')
plt.title('Percentage of Home Loans by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Percentage of Home Loans (%)')
plt.xticks(rotation=45)
plt.show()


# In[26]:


# Calculate mean and median annual income for each age group
income_stats = df.groupby('age_group')['Annual Income'].agg(['mean', 'median'])
print(income_stats)


# In[27]:


# Scatter plot of annual income vs. age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='Annual Income', data=df, alpha=0.6, edgecolor=None)
plt.title('Annual Income vs. Age')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.show()


# In[28]:


# Box plot of annual income by age group
plt.figure(figsize=(12, 6))
sns.boxplot(x='age_group', y='Annual Income', data=df, palette='viridis')
plt.title('Distribution of Annual Income by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Annual Income')
plt.xticks(rotation=45)
plt.show()


# In[29]:


# Calculate correlation coefficient between age and annual income
correlation = df[['age', 'Annual Income']].corr().iloc[0, 1]
print(f"Correlation coefficient between age and annual income: {correlation:.2f}")


# In[30]:


# For Categorical Variables
from scipy.stats import chi2_contingency

# Example categorical variables: 'job', 'marital', 'contact', 'housing', 'loan'

# Function to perform Chi-Square test
def chi2_test(data, feature, target):
    contingency_table = pd.crosstab(data[feature], data[target])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return chi2, p

# Perform Chi-Square Test for each categorical variable
categorical_features = ['job', 'marital', 'contact', 'housing', 'loan']
for feature in categorical_features:
    chi2, p = chi2_test(df, feature, 'Term Deposit')
    print(f'Feature: {feature}, Chi-Square: {chi2:.2f}, p-value: {p:.4f}')


# In[31]:


# For Numerical Variables
import numpy as np
from scipy.stats import pearsonr, f_oneway
# For numerical variables: 'age', 'annual_income', 'balance'
# Correlation with numerical variables
numerical_features = ['age', 'Annual Income', 'balance']
for feature in numerical_features:
    corr, _ = pearsonr(df[feature], df['Term Deposit'].apply(lambda x: 1 if x == 'yes' else 0))
    print(f'Feature: {feature}, Pearson Correlation: {corr:.2f}')
# ANOVA for each numerical variable by Term Deposit status
for feature in numerical_features:
    grouped = df.groupby('Term Deposit')[feature].apply(list)
    f_statistic, p_value = f_oneway(*grouped)
    print(f'Feature: {feature}, ANOVA F-Statistic: {f_statistic:.2f}, p-value: {p_value:.4f}')


# In[32]:


# One-Hot Encoding for categorical columns
df_encoded = pd.get_dummies(df, columns=['Insurance', 'housing', 'loan', 'contact', 'poutcome', 'job', 'marital', 'education', 'Gender'], drop_first=True)

# Verify the changes
print(df_encoded.head())


from sklearn.preprocessing import LabelEncoder
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Label encode the target variable
df_encoded['Term Deposit'] = label_encoder.fit_transform(df['Term Deposit'])

# Verify the changes
print(df_encoded[['Term Deposit']].head())

# Check for unique categories in 'age_group'
print(df['age_group'].unique())

# If you decide to use one-hot encoding for 'age_group'
df_encoded = pd.get_dummies(df_encoded, columns=['age_group'], drop_first=True)


# In[33]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Label encode the target variable 'Term Deposit'
label_encoder = LabelEncoder()
df['Term Deposit'] = label_encoder.fit_transform(df['Term Deposit'])
# One-Hot Encoding for categorical columns
df_encoded = pd.get_dummies(df, columns=['Insurance', 'housing', 'loan', 'contact', 'poutcome', 'job', 'marital', 'education', 'Gender'], drop_first=True)

# If 'age_group' is to be included, perform one-hot encoding
if 'age_group' in df_encoded.columns:
    df_encoded = pd.get_dummies(df_encoded, columns=['age_group'], drop_first=True)

# Separate features and target variable
X = df_encoded.drop(columns=['Term Deposit', 'Customer_number'])  # Drop target and non-feature columns
y = df_encoded['Term Deposit']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC

# Evaluate the model
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("Test ROC AUC Score:")
print(roc_auc_score(y_test, y_test_prob))


# In[34]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[35]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Train-Test Split (80:20)pro
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and fit the model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predict on test set
y_pred_gb = gb_model.predict(X_test)
y_proba_gb = gb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))
print("Gradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))
print("Gradient Boosting ROC AUC Score:", roc_auc_score(y_test, y_proba_gb))


# In[36]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba_gb)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_gb)
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[38]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_gb)
# Compute average precision score
avg_precision = average_precision_score(y_test, y_proba_gb)
# Plot Precision-Recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {avg_precision:.2f})')
plt.grid(True)
plt.show()


# In[41]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
get_ipython().system('pip install xgboost')
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("Random Forest ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))


# In[42]:


xgb_model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))
print("XGBoost ROC AUC Score:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))


# In[ ]:




