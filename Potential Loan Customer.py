#!/usr/bin/env python
# coding: utf-8

# # <center> Logistic Regression

# ## <center> Case-Study- Predict potential Loan Customers

# <img src="https://s3.ap-south-1.amazonaws.com/img1.creditmantri.com/community/article/home-loan-charges-that-every-home-loan-applicant-should-be-aware-of.jpg" >

# # <center> Problem Description

# We have data from Zenith bank that has a growing customer base. A bank has 2 types of customers :1. Liability Cutomers : Those who have deposits with the bank 2. Asset Customers    : Those who have a loan with the bank
#     
# For Zenith bank, the majority of customers are liability customers currently (depositors). They are struggling with lower ratio of borrowers (Assets). To address this problem, they are going to run a huge camapign with a lot of Marketing Spend to increase the borrowers. This is where Machine Learning can help. The request is to provide them with a ML model that can guide them on which customers to target so that their spends are efficiently utilised. 
# 
# 
# As a Data scientist, we have to build a model for Zenith Bank that will ease the task of marketing department by predicting most likely borrowers</p>
#   
#     
# Main Objective of this exercise : To understand Logistic regression and explore this algorithm using Sklearn, Statmodel, and related concepts like Roc-Auc Curve, Coefficients, Feature Selection etc
# 
# **Questions to be answered-**
# * Which customers will borrow a loan?
# * From the given data, which features are most significant in determining above.
# 

# # <center> Importing Libraries & Data

# In[1289]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

pd.set_option('display.max_columns',250)
pd.set_option('display.max_rows',250)

from IPython.display import display


# In[1290]:


original_data=pd.read_csv('zenith_bank_customer_loans.csv')
customer_data=original_data.copy()
print("We have {0} rows and {1} columns in our data".format(customer_data.shape[0],customer_data.shape[1]))


# In[1291]:


customer_data.head()


# #  Data Dictionary

#     
# * ID: Unique ID of Customer
# * Age: Age of the Customer
# * Experience: Years of experience
# * Income: Annual income `(in $1000)`
# * ZIP Code: ZIP code of residence
# * Family: Count of family members
# * CCAvg: Avg Monthly Spending on Credit Card `(in $1000)`
# * Education: 1- Undergrad; 2- Graduate;3- Advanced/Professional # can be used as education_Score
# * Mortgage: House Mortgage (in $1000)
# * Personal_Loan: Flag indicating whether customer opted for loan previously or not ( 1 - Opted for loan)
# * Securities_Account: Does the customer have securities account with the bank?
# * CD_Account: Flag indicating whether the customer has a certificate of deposit (CD) account with the bank
# * Online: Flag indicating whether the customer use internet banking facilities
# * CreditCard: Flag indicating whether the customer use a credit card issued by any other Bank (excluding Zenith Bank)?

# In[1292]:


customer_data.columns


# # <center> Preprocessing Data

# ## Basic Data Check

# ***checking for missing values***

# In[1293]:


customer_data.isna().sum().sum()


# **Number of unique values in each column**

# It can help us in detemiming categorical/continous features

# In[1294]:


customer_data.nunique()


# In[1295]:


customer_data.info()


# In[1296]:


customer_data.sample(15)


# In[1297]:


# dropping Id


# In[1298]:


customer_data.drop('ID',axis=1,inplace=True)


# ### Renaming columns

# Removing whitespaces

# In[1299]:


customer_data.rename(columns={"ZIP Code":"ZIPCode","Personal Loan":"PersonalLoan",
                               "Securities Account":"SecuritiesAccount",
                               "CD Account":'CDAccount'},inplace=True)


# ### Preprocessing zipcode

# using zipcode library to map zipcodes to countries.

# In[1300]:


customer_data.ZIPCode.nunique()


# In[1301]:


#intallling zipcode library

get_ipython().system('pip install zipcodes')


# In[1302]:


# list of unique zipcodes
zipcodes=customer_data.ZIPCode.unique()


# In[1303]:


zipcodes[:5]


# In[1304]:


import zipcodes as zcode
zip_dict={}

for i in zipcodes:
    city_country=zcode.matching(i.astype('str'))
    
    if len(city_country)==1:
        country=city_country[0].get('county')
        
    else:
        country=i
    
    zip_dict.update({i:country})


# In[1309]:


zip_dict


# In[1311]:


import numbers

l1=list(zip_dict.values())
[i for i in l1 if isinstance(i, numbers.Number)]


# In[1312]:


# our mapping was able to map all the zipcodes except - [92717, 9307, 92634, 96651]
# as these are only few, we can fix them mannualy taking refrences from the internet


# In[1313]:


zip_dict.update({92717:'Orange County'})
zip_dict.update({92634:'Orange County'})
zip_dict.update({96651:'Orange County'})
zip_dict.update({9307:'Orange County'})


# In[1314]:


# we can map countries and add to the origial data


# In[1315]:


customer_data['country']=customer_data.ZIPCode.map(zip_dict)


# In[1316]:


customer_data.head()


# In[1317]:


customer_data.country.nunique()


# In[1318]:


customer_data.info()


# In[1319]:


# some of the categorical varibles are stored as int64 which takes lot of space.
# changing thier datatype to categories instead


# In[1320]:


customer_data.describe()


# In[1321]:


# minimum_values of experience is -ve which seems incorrect


# In[1322]:


# categorical variables like Education,Mortgage,PersonalLoan,SecuritiesAccount,CDAccount,Online,CreditCard,ZIPCode,Family


# In[1323]:


cat_features=['PersonalLoan', 'SecuritiesAccount','Family', 'CDAccount', 'Online', 'CreditCard', 'ZIPCode', 'Education','country']
customer_data[cat_features]=customer_data[cat_features].astype('category')


# In[1324]:


customer_data.info()


# In[1325]:


# we can see tha he memory usage has been decreased from 547 to 266.4


# **Handling experience**

# In[714]:


customer_data[customer_data['Experience']<0].head()


# In[715]:


customer_data[customer_data['Experience']<0].Age.describe()


# So there are 52 instances of -ve experience , having age from 23-29.

# In[716]:


customer_data.groupby(['Age','Education'])['Experience'].describe()


# As per education & age it is evident that values are negative by mistake.Hence,
# converting them into absolute values.

# In[717]:


customer_data['Experience']=abs(customer_data['Experience'])


# In[718]:


customer_data.describe()


# # <center> EDA

# ### EDA| 5 Point summar

# In[719]:


customer_data.describe()


# * The experience seems to be fine noe,as minimum experience value is 0
# * Age ranges from 23-67, and mean age is 45
# * Income ranges from 8 grand to 224 grand , with mean at 73
# * CCavg seems to be in between 0-10 grand,with avg customer spending $1.93.
# * Mortgage range is 0-635 grand, while mean is around 56.

# In[720]:


for i in cat_features:
    display(customer_data[i].value_counts(normalize=True)*100)


# 22% customers belong to Los Angeles County.
# only about 9.6% had borrowed loan previously.

# In[721]:


customer_data.isna().sum().sum()


# In[722]:


data=customer_data


# In[723]:


data.isna().sum().sum()


# ## EDA| Univariate Analysis-continous variables

# In[724]:


num_f =  ['Age','Experience','Income','CCAvg','Mortgage']


# In[725]:


a=0
for i in num_f:
    sns.set_theme(style='white')
    plt.figure(figsize=(8,10))
    plt.subplot(len(num_f)*2,1,a+1)
    plt.title(" Box plot of " +i)
    sns.boxplot(x=data[i],orient='h',color='violet')
    sns.despine(top=True,right=True,left=True)
    plt.figure(figsize=(8,32))
    plt.subplot(len(num_f)*2,1,a+2)
    plt.title("Dist-plot "+ i)
    sns.distplot(x=data[i],kde=False,color='blue')
    ax_distplot=sns.distplot(x=data[i],kde=False,color='blue')
    sns.despine(top=True,right=True,left=True)
    a+=2
    
    mean=data[i].mean()
    median=data[i].median()
    mode=data[i].mode().tolist()[0]
    ax_distplot.axvline(mean, color='r', linestyle='--',linewidth=2)              
    ax_distplot.axvline(median, color='g', linestyle='-',linewidth=2)               
    ax_distplot.axvline(mode, color='y', linestyle='-',linewidth=2)
    
    plt.legend({'Mean':mean,'Median':median,'Mode':mode})
    


# **Observations**

# * Age and experience have similar distribution
# * Income, Mortgage & CCAvg all are Righlt skewed
# * Mortgage is 0 for a lot of customers

# ## EDA| Univariate Analysis- Categorical variables

# In[635]:


num_f=data.columns[~data.columns.isin(num_f)]


# In[728]:


num_f


# In[729]:


data.Age.nunique()


# In[730]:


data.Age.describe()


# since age has 45 unique values , to see a pattern it makes sense to conver them into bins

# In[731]:


data['Age_bins']=pd.cut(data.Age,bins=[0,30,40,50,60,100],
                        labels=['18-30','31-40','41-50','51-60','61-100'])


# In[732]:


data.Income.nunique()


# In[733]:


data.Income.describe() 


# In[734]:


data['Income_segment']=pd.cut(data.Income,bins=[0,50,150,250],
                            labels=["Lower",'Middle','High'])


# In[735]:


data.CCAvg.nunique()


# In[738]:


data.CCAvg.describe() 


# In[743]:


data['Spending_segment']=pd.cut(data.CCAvg,bins=[0.00000, 0.70000, 2.50000, 10.00000],
                               labels=["Low", "Medium", "High"],include_lowest=True)


# In[744]:


data.isna().sum().sum()


# **Univariate Analysis-Categorical Variables(using countplot)**

# In[653]:


plt.figure(figsize=(15,20))
sns.set_theme(style="white");


# In[654]:


data.country.value_counts()


# In[655]:


#we will analyze counties later on , as neither we can
# bin them nor are they sufficiently less to draw charts


# In[656]:


cat_cols = ['Family','Education','PersonalLoan','SecuritiesAccount',
               'CDAccount','Online','CreditCard',
               'Age_bins','Income_segment','Spending_segment']


# In[657]:


sns.set_theme(style='white');


# In[1037]:


plt.figure(figsize=(15,20))
titles=['Family Members Count','Education Profile','Customers with/without Previous Loan',
       'Customers with/without Security Account','Customers with/without CD Account',
       'Customers with/without Online banking',' Customers with/without Credit Card',
       'Age bins',"Income group",'Spending group']
for i,variable in enumerate(cat_cols):
    plt.subplot(5,2,i+1)
    sns.countplot(x=data[variable],data=data)
    sns.set_palette('Set2')
    sns.despine(top=True,right=True,left=True)
    plt.title(titles[i],size=18,color='blue')
    plt.tight_layout()
    perc=data[variable].value_counts(normalize=True)*100
    for x,y in enumerate(perc):
         plt.annotate(str(round(y,2))+'%', xy=(x, y), ha='center',fontsize='15')
    plt.tight_layout()


# **Univariate Analysis-Countries**

# In[659]:


data.groupby(['country','PersonalLoan'])['PersonalLoan'].count().unstack().head()


# In[666]:


#plotting theis data via Crosstab


# In[661]:


pd.crosstab(index=data.country,columns=data.PersonalLoan.sort_values(ascending=False)).plot(kind='barh',stacked=True,figsize=[20,15])
plt.tight_layout()


# **Countries are an importnat features since loans vary a lot by countries.But countries are a lot in number and this level i too granular to analyze.Hence  we can map these regions using information from internet.**
# 
# Source used for below step - 
# https://www.calbhbc.org/region-map-and-listing.html

# In[662]:


counties_region_mapping = {
'Los Angeles County':'Los Angeles Region',
'San Diego County':'Southern',
'Santa Clara County':'Bay Area',
'Alameda County':'Bay Area',
'Orange County':'Southern',
'San Francisco County':'Bay Area',
'San Mateo County':'Bay Area',
'Sacramento County':'Central',
'Santa Barbara County':'Southern',
'Yolo County':'Central',
'Monterey County':'Bay Area',            
'Ventura County':'Southern',             
'San Bernardino County':'Southern',       
'Contra Costa County':'Bay Area',        
'Santa Cruz County':'Bay Area',           
'Riverside County':'Southern',            
'Kern County':'Southern',                 
'Marin County':'Bay Area',                
'San Luis Obispo County':'Southern',     
'Solano County':'Bay Area',              
'Humboldt County':'Superior',            
'Sonoma County':'Bay Area',                
'Fresno County':'Central',               
'Placer County':'Central',                
'Butte County':'Superior',               
'Shasta County':'Superior',                
'El Dorado County':'Central',             
'Stanislaus County':'Central',            
'San Benito County':'Bay Area',          
'San Joaquin County':'Central',           
'Mendocino County':'Superior',             
'Tuolumne County':'Central',                
'Siskiyou County':'Superior',              
'Trinity County':'Superior',                
'Merced County':'Central',                  
'Lake County':'Superior',                 
'Napa County':'Bay Area',                   
'Imperial County':'Southern',
93077:'Southern',
96651:'Bay Area'
}


# In[753]:


data['Regions']=data.country.map(counties_region_mapping)


# In[754]:


data.Regions.nunique()


# In[755]:


data.isna().sum().sum()


# In[756]:


#hence no null


# **Univariate Analysis-Region**

# In[764]:


plt.figure(figsize=(10,5))
sns.countplot(x=data.Regions,data=data)
sns.despine(top=True,right=True,left=True)


# * Bay Area has max customers followed by southern region
# *Superior region has very customers

# ## EDA|Bivariate & Multivariate Analysis

# ### <center> Heatmap

# In[769]:


plt.figure(figsize=(10,5))
sns.heatmap(data=data.corr(),annot=True,linecolor='white');


# ### Bivariate Analysis- Pairplot

# In[777]:


sns.pairplot(data=data,hue='PersonalLoan',diag_kind=None,palette='Set2',corner=True);


# **heatmap and pairplot- insights**

# * Age & Experience seems to be highly correlated
# * Income and Average spendings are positively correlated
# * Opposite to our intiution ,Mortage has very little correlation with income
# * Higher income & higher spending customers are contributing a lot to potential buyers

# **Bivariate Analysis - Boxplot comparison of Numerical features v/s Target**

# In[784]:


num_f


# In[804]:


plt.figure(figsize=(15,8))
for i,variable in enumerate(num_f):
    plt.subplot(2,3,i+1)
    sns.boxplot(x='PersonalLoan',y=data[variable],data=data,palette='Set2')
    sns.despine(left=True,right=True,top=True)
    plt.tight_layout()


# *Insighst-* 
# * Age & Target doesn't seem to impact our target variable
# * Same story with the experience
# * Higher income customer and heigher spendings are more likely to take the loan
# * Those taking loans, have higher mortgage

# **Bivariate Analsysis- Distplot comparision of CCAvg v/s Target**

# In[828]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
sns.distplot(data[data.PersonalLoan==1].CCAvg, color='g', label='Personal Loan = 1')
sns.distplot(data[data.PersonalLoan==0].CCAvg, color='r', label='Personal Loan = 0')
plt.xlabel('Credit Card Spending (in thousands)')
plt.ylabel('Density')
plt.title('Distribution of Credit Card Spending by Personal Loan Status')
plt.legend()
sns.despine(top=True, right=True, left=True)
plt.show()


# In[836]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
sns.distplot(data[data.PersonalLoan==1].Income, color='g', label='Personal Loan = 1')
sns.distplot(data[data.PersonalLoan==0].Income, color='r', label='Personal Loan = 0')
plt.xlabel('Income (in thousands)')
plt.ylabel('Density')
plt.title('Distribution of Income by Personal Loan Status')
plt.legend()
sns.despine(top=True, right=True, left=True)


# In[837]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
sns.distplot(data[data.PersonalLoan==1].Mortgage, color='g', label='Personal Loan = 1')
sns.distplot(data[data.PersonalLoan==0].Mortgage, color='r', label='Personal Loan = 0')
plt.xlabel('Mortgage (in thousands)')
plt.ylabel('Density')
plt.title('Distribution of Mortgage by Personal Loan Status')
plt.legend()
sns.despine(top=True, right=True, left=True)


# In[838]:


plt.figure(figsize=(8,4))
sns.distplot(data[data.PersonalLoan==1].Age,color='g',label='Personal Loan = 1')
sns.distplot(data[data.PersonalLoan==0].Age,color='r',label='Personal Loan = 1')
sns.despine(top=True,right=True,left=True)
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.tight_layout();


# ### Distplot Comparision Insight

# * The distribution of personal_loan=1 is left skewed for income

# **Bivariate Analysis| Stacked Bars- Categorical Features(s) v/s Target**

# In[859]:


cat_cols=['Family','Education','SecuritiesAccount','CDAccount','CreditCard','Online','Regions','Age_bins',
             'Income_segment','Spending_segment']
cat_f=cat_cols


# In[972]:


for i,variable in enumerate(cat_f):
    df=pd.crosstab(data[variable],data['PersonalLoan'],normalize='index')
    print(df)
    print('*******************')
    print('*******************')
    print('*******************')
    print('*******************')
    sns.set_palette('Set2')
    df.plot(kind='bar',stacked=True)
    plt.xticks(rotation='0')
    sns.despine(left=True,right=True,top=True)
    plt.legend(labels=["No","Yes"],bbox_to_anchor=(1,1))
    plt.show();


# ### Multivariate Analysis

# *Multivariate Analysis|Income & Average spend across Target*

# In[981]:


plt.figure(figsize=(6,10))
sns.relplot(x='Income_segment',y='CCAvg',hue='PersonalLoan',data=data);


# **Multivariate Analysis| Income & Mortgage- across Target**

# In[994]:


sns.swarmplot(x='Income_segment',y='Mortgage',hue='PersonalLoan',data=data)
sns.despine(top=True,right=True,left=True)


# ##### Multivariate Analysis |  Income & Education - across Target

# In[989]:



sns.swarmplot(x='Education',y='Income',hue='PersonalLoan',data=data)
sns.despine(top=True,right=True,left=True)
plt.legend(labels=["No","Yes"],title="Borrowed Loan ? ",
           bbox_to_anchor=(1,1), colors=["blue", "green"])


# **Multivariate Analysis| Income V/S Agebins V/S Education - across Targets (0,1)**

# In[1003]:


sns.catplot(x='Age_bins',y='Income',hue='Education',kind='bar',data=data,col='PersonalLoan');
sns.despine(top=True,right=True,left=True);


# **Multivariate Analysis| Education V/S Average_spend - across Targets (0,1)**

# In[1017]:


sns.set_palette('Set2')
sns.catplot(x='Education',y='CCAvg',hue='PersonalLoan',kind='bar',data=data);
sns.despine(top=True,right=True,left=True);


# **Multivariate Analysis | insight**

# * Higher income is a major factor in detemining the potential loan borrowers.
# * HeigherMortgage also drive loan to an an extent.
# * Higher average spends are also linked to loan borrowers.
# * Income,Average spends and mortgage all seems to be correlated.

# **EDA|Checking imbalance in Target column**

# In[1026]:


plt.pie(x=data.PersonalLoan.value_counts(),autopct='%1.2f%%',labels=['No_Loan','Loan'],labeldistance=1.1);


# **The data seems be imbalanced with respect to Target variable,which is not a good indication.It might create a baised model,or model that is more accurate for No_loan**

# We will use SMOTE later to rectify this

# In[ ]:





# ## <center> Insight Summary

# **Knowing data :**
# 
# * Target column is PersonalLoan(categorical datatype).
# * We have following continous features- Age, Experience,Income,Mortgage,CCAvg.
# * No missing values in our data now.

# **Data Cleaning :**
# 
# * Experience had some -ve values which we treated by taking absolutes.
# * In order to reduce granulaity,Zipcodes were mapped to countries using Zipcode library.And Countries were further mapped to Regions using information fro internet.
# * Age,Income & Spendings were used to derive corresponding categories for better analysis.

# **Insights :**

# * Higher Income, Spend and Mortgages are indicator of loan borrowers customer segment. 
# * Family size of 3 came out to be the best in terms of loan customers.
# * Education level 2: Graduate and 3: Advanced/Professional  were also good indicators of our desired segment.
# * Certificate of deposit is a slightly positive indicator for a loan borrower.
# * Customers with other bank Credit card showed no corelation with loans.
# * LosAngeles and Bay region contribute maximum towards loan borrowers.
# * All age segments are equally likely for loan customers.
# * It can be inferred that the more income you have, the more you tend to spend and live a "larger than life" lifestyle.

# **Tenatative profiling of a customer**

# * Most Probable Borrowers: Customer with high income,spending and mortgage.Also,which have a security deposit.
# * Likel Borrowers: Customers in medium income range,average spends and mortgages.
# * Unlikely Borrowers: Rest of the customers

# **Next step for data-preprocessing**

# * Since we already know the correlations, based on that Experience can be dropped as it is very highly correlated to Age.
# * We can also drop following logical duplicate features which we derived from EDA:-> Experience,Country,Zipcode,Agebins,Income_segment & Spending_segmen 

# ## <center> Feature Engineering

# ###  Feature Engineering | Dropping Unnecessary or duplicate features

# In[1124]:


data.info()


# In[1126]:


# As per EDA, dropping columns which are either correlated with some feature or are duplicates
data.drop(columns=["Age_bins", "ZIPCode","country",'Experience','Income_segment','Spending_segment'], inplace=True)


# In[1127]:


data.info()


# **Creating dummies**

# In[1160]:


data[data.columns[data.columns.isin(cat_f1)]]


# In[1162]:


# here we can see that our 4 cat_f are already in binary;namely 
# SecuritiesAccount,CDAccount,Online,CreditCard


# In[1163]:


X=data.drop('PersonalLoan',axis=1)
Y=data['PersonalLoan']


# In[1164]:


X=pd.get_dummies(data=X,columns=['Regions','Education'],drop_first=True)


# **Train_Test_Split**

# In[1166]:


from sklearn.model_selection import train_test_split


# In[1168]:


X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)


# **Standardization**

# In[1170]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()

X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)

X_train_std=pd.DataFrame(X_train_std,columns=X_train.columns)
X_test_std=pd.DataFrame(X_test_std,columns=X_test.columns)


# # <center>  Logistic Regression

# **There can be two types of errors our model can make**

# 1. Type 1 Error : Model predicts customer will opt for loan but he/she doesn't(Loss of Resources)
# 2. Type 2 Error : Model predicts customer will not opt for loan but he/she actually wanted to. (Opportunity loss)

# **Which of the above is more important**
# 
# * The main objective of this project is to acquire more customer,hence TYPE-2 error is more critical.It is okay if the marketing budget is spent somewhat incorrectly but we should not miss out the opportunity against a  potential customer.
# * For this reason, Recall is our right matrix to predict performance.Accuracy could have been used, but since our data is highly baised,it would not be a very good choice.

# In[1235]:


from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(class_weight='weighted')
model=log_reg.fit(X_train_std,y_train)

pred_train=model.predict(X_train_std)
pred_test=model.predict(X_test_std)


# In[1236]:


# When we set class_weight='weighted' in LogisticRegression, 
# the algorithm automatically adjusts the weight of each class 
# to be inversely proportional to the class frequencies in the input data.
# This means that the algorithm will give more importance to the minority class
# during training, which can help improve the model's ability to predict the minority class.

# However, it's important to note that using class_weight='weighted' 
# may not always be as effective as using SMOTE, 
# especially if the class imbalance is severe.
# In such cases, using a combination of both techniques may be 
# necessary to achieve better results.


# In[1237]:


# importing confusion matrix
from sklearn.metrics import confusion_matrix

train_confusion_matrix=confusion_matrix(y_train,pred_train)
test_confusion_matrix=confusion_matrix(y_test,pred_test)


# In[1238]:


display(train_confusion_matrix,test_confusion_matrix)


# **Checking ACCURACY,PRECISION,RECALL,F-1 SCORE**

# In[1239]:


from sklearn.metrics import classification_report


# In[1240]:


training_report=classification_report(y_train,pred_train)
test_report=classification_report(y_test,pred_test)


# In[1241]:


print(training_report)

print('********************************************************')
print(test_report)
print('********************************************************')
print("Suport indicates no of columns in each case")


# **The recall of the minority class in very less. It proves that the model is more biased towards majority class. So, it proves that this is not the best model.
# Now, we will apply different imbalanced data handling techniques and see their accuracy and recall results.**

# ## Using SMOTE algorithm

# In[1263]:


print("Before oversampling , count of label 0 in training : {0}".format(sum(y_train==0)))
print("Before oversampling , count of label 1 in training : {0}".format(sum(y_train==1)))


# In[1272]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()

X_train_res,y_train_res=sm.fit_resample(X_train_std,y_train)


# In[1275]:


print("After oversampling , count of label 0 in training : {0}".format(sum(y_train_res==0)))
print("After oversampling , count of label 1 in training : {0}".format(sum(y_train_res==1)))


# In[1276]:


# Again making maodel with the non baised data. 


# In[1280]:


from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(class_weight='weighted')
model=log_reg.fit(X_train_res,y_train_res)

pred_train=model.predict(X_train_res)
pred_test=model.predict(X_test_std)


# In[1281]:


# importing confusion matrix
from sklearn.metrics import confusion_matrix

train_confusion_matrix=confusion_matrix(y_train_res,pred_train)
test_confusion_matrix=confusion_matrix(y_test,pred_test)


# In[1282]:


display(train_confusion_matrix,test_confusion_matrix)


# In[1285]:


from sklearn.metrics import classification_report


# In[1286]:


training_report=classification_report(y_train_res,pred_train)
test_report=classification_report(y_test,pred_test)


# In[1287]:


print(training_report)

print('********************************************************')
print(test_report)
print('********************************************************')
print("Suport indicates no of columns in each case")


# # Recall value of minority class has improved to 92 %
