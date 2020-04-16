#All packages:

import numpy as np
import pandas as pd
#Cause 'seaborn' is based on matplotlib:
import matplotlib.pyplot as mplt
#For statistical data visualization:
#import seaborn as sns
#All the following functionalities have been imported for machine learning, data splitting, accuracy testing:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#For scaling the unscaled data: (to avoid biasing?):
from sklearn.preprocessing import StandardScaler
#I can probably manually encode, I am just bored...:
from sklearn.preprocessing import LabelEncoder
#for Random Forest in a regression approach:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
#For random forest classification:
from sklearn.ensemble import RandomForestClassifier
#Calculating the accuracy of Classifier:
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#For Naive Bayes:
from sklearn.naive_bayes import GaussianNB
#For measuring the performance in terms of time taken:
import time

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

def outlier_test(inp):
    #I am using the condiiton of absolute value of z-score>3 as the parameter of being an outlier:
    threshold=3
    avg_inp=np.mean(inp)
    stdev=np.std(inp)
    zscores=[]
    for i in inp:
        z=(i-avg_inp)/stdev
        zscores.append(z)
    return np.where(np.abs(zscores) > threshold)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#Function for random forest regressor:

def forest_regressor(for_for):

    #--------------------------------------------------------------------------------------------------------

    start=time.time()

    #--------------------------------------------------------------------------------------------------------

    x_for=for_for.iloc[:,0:9].values
    y_for=for_for.iloc[:,9].values

    x_for_train,x_for_test,y_for_train,y_for_test=train_test_split(x_for,y_for,test_size=0.2,random_state=0)

    #--------------------------------------------------------------------------------------------------------

    #Let's start with, encoding:

    lab_encoder=LabelEncoder()
    #encode=["Gender","City_Category"]
    #for col in encode:
    #For Gender:
    x_for_train[:,0]=lab_encoder.fit_transform(x_for_train[:,0])
    x_for_test[:,0]=lab_encoder.fit_transform(x_for_test[:,0])

    #For Age Groups:
    x_for_train[:,1]=lab_encoder.fit_transform(x_for_train[:,1])
    x_for_test[:,1]=lab_encoder.fit_transform(x_for_test[:,1])

    #For City Category:
    x_for_train[:,3]=lab_encoder.fit_transform(x_for_train[:,3])
    x_for_test[:,3]=lab_encoder.fit_transform(x_for_test[:,3])

    #--------------------------------------------------------------------------------------------------------

    #Now we gonna do scaling:
    scalar=StandardScaler()
    x_for_train=scalar.fit_transform(x_for_train)

    test_scalar=StandardScaler()
    x_for_test=test_scalar.fit_transform(x_for_test)

    #--------------------------------------------------------------------------------------------------------

    end=time.time()
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Time taken for pre-processing for the Random Forest Regressor is:",end-start,"seconds")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #--------------------------------------------------------------------------------------------------------

    max_leaves=input("Maximum number of leaf nodes you wish to have: ")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    max_leaves=int(max_leaves)

    #--------------------------------------------------------------------------------------------------------

    start=time.time()

    #--------------------------------------------------------------------------------------------------------

    model=RandomForestRegressor(max_leaf_nodes=max_leaves,random_state=0)
    model.fit(x_for_train,y_for_train)
    pred=model.predict(x_for_test)

    #--------------------------------------------------------------------------------------------------------

    end=time.time()
    print("Time taken for Random Forest Regressor to execute:",end-start,"seconds.")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #--------------------------------------------------------------------------------------------------------

    mae=mean_absolute_error(y_for_test,pred)
    print("For",max_leaves,"nodes, the mean absolute error:",mae)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    mse=mean_squared_error(y_for_test,pred)
    rmse=np.sqrt(mse)
    print("Root Mean Squared value for",max_leaves,"nodes is:",rmse)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#Function for random forest classifier:

def forest_classifier(for_for):

    #--------------------------------------------------------------------------------------------------------

    #We gotta make bins for purchase amount:
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    num_bins=input("Enter the number of bins: ")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    num_bins=int(num_bins)

    #--------------------------------------------------------------------------------------------------------

    start=time.time()

    #--------------------------------------------------------------------------------------------------------

    bin_width=(((max(for_for.Purchase))-(min(for_for.Purchase)))/num_bins)
    print("For",num_bins,"bins, bin width=",bin_width)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    bins=[]
    lab=[]
    i=0
    while i<num_bins+1:
        if i==0:
            bins.append(min(for_for.Purchase))
            lab.append(i)
        else:
            bins.append(bins[i-1]+bin_width)
            lab.append(i)
        i=i+1

    #print(bins)
    lab=lab[:-1]
    #print(lab)

    #--------------------------------------------------------------------------------------------------------

    x_for=for_for.iloc[:,0:9].values
    y_for=for_for.iloc[:,9].values

    x_for_train,x_for_test,y_for_train,y_for_test=train_test_split(x_for,y_for,test_size=0.2,random_state=0)

    #--------------------------------------------------------------------------------------------------------

    y_train_bin=pd.cut(y_for_train,bins=bins,labels=lab)
    print(y_train_bin[:20])
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    y_test_bin=pd.cut(y_for_test,bins=bins,labels=lab)

    y_train_bin=y_train_bin.fillna(num_bins-1)
    y_test_bin=y_test_bin.fillna(num_bins-1)

    #--------------------------------------------------------------------------------------------------------

    #Let's start with, encoding:

    lab_encoder=LabelEncoder()
    #encode=["Gender","City_Category"]
    #for col in encode:
    #For Gender:
    x_for_train[:,0]=lab_encoder.fit_transform(x_for_train[:,0])
    x_for_test[:,0]=lab_encoder.fit_transform(x_for_test[:,0])

    #For Age Groups:
    x_for_train[:,1]=lab_encoder.fit_transform(x_for_train[:,1])
    x_for_test[:,1]=lab_encoder.fit_transform(x_for_test[:,1])

    #For City Category:
    x_for_train[:,3]=lab_encoder.fit_transform(x_for_train[:,3])
    x_for_test[:,3]=lab_encoder.fit_transform(x_for_test[:,3])

    #--------------------------------------------------------------------------------------------------------

    #Now we gonna do scaling:
    scalar=StandardScaler()
    x_for_train=scalar.fit_transform(x_for_train)

    test_scalar=StandardScaler()
    x_for_test=test_scalar.fit_transform(x_for_test)

    #--------------------------------------------------------------------------------------------------------

    end=time.time()
    print("Time taken for pre-processing for the Random Forest Classifier is:",end-start,"seconds")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #--------------------------------------------------------------------------------------------------------

    max_leaves=input("Maximum number of leaf nodes you wish to have: ")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    max_leaves=int(max_leaves)

    #--------------------------------------------------------------------------------------------------------

    start=time.time()

    #--------------------------------------------------------------------------------------------------------

    classifier=RandomForestClassifier(max_leaf_nodes=max_leaves,bootstrap=True,criterion='gini',n_estimators=150)
    classifier.fit(x_for_train,y_train_bin)
    pred=classifier.predict(x_for_test)

    #--------------------------------------------------------------------------------------------------------

    end=time.time()
    print("Time taken for execution of the Random Forest Classifier is:",end-start,"seconds")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #--------------------------------------------------------------------------------------------------------

    print("The confusion matrix for the Random Forest Classifier is as follows:")
    print(confusion_matrix(y_test_bin,pred))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("The classification report for the Random Forest Classifier is as follows:")
    print(classification_report(y_test_bin,pred))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Accuracy for the Random Forest Classifier model:",(accuracy_score(y_test_bin,pred))*100,"%")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#Accuracy for gaussian Naive Bayes, 63.23% for 3 bins, plummets down to 45.15% when bins go to 4.

#For Gaussian Naive Bayes:

def naive_bayes(for_for):

    #--------------------------------------------------------------------------------------------------------

    start=time.time()

    #--------------------------------------------------------------------------------------------------------

    #We gotta make bins for purchase amount:
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    num_bins=input("Enter the number of bins: ")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    num_bins=int(num_bins)

    bin_width=(((max(for_for.Purchase))-(min(for_for.Purchase)))/num_bins)
    print("For",num_bins,"bins, bin width=",bin_width)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    bins=[]
    lab=[]
    i=0
    while i<num_bins+1:
        if i==0:
            bins.append(min(for_for.Purchase))
            lab.append(i)
        else:
            bins.append(bins[i-1]+bin_width)
            lab.append(i)
        i=i+1

    #print(bins)
    lab=lab[:-1]
    #print(lab)

    #--------------------------------------------------------------------------------------------------------

    x_for=for_for.iloc[:,0:9].values
    y_for=for_for.iloc[:,9].values

    x_for_train,x_for_test,y_for_train,y_for_test=train_test_split(x_for,y_for,test_size=0.2,random_state=0)

    #--------------------------------------------------------------------------------------------------------

    y_train_bin=pd.cut(y_for_train,bins=bins,labels=lab)
    print(y_train_bin[:20])
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    y_test_bin=pd.cut(y_for_test,bins=bins,labels=lab)

    y_train_bin=y_train_bin.fillna(num_bins-1)
    y_test_bin=y_test_bin.fillna(num_bins-1)

    #--------------------------------------------------------------------------------------------------------

    #Let's start with, encoding:

    lab_encoder=LabelEncoder()
    #encode=["Gender","City_Category"]
    #for col in encode:
    #For Gender:
    x_for_train[:,0]=lab_encoder.fit_transform(x_for_train[:,0])
    x_for_test[:,0]=lab_encoder.fit_transform(x_for_test[:,0])

    #For Age Groups:
    x_for_train[:,1]=lab_encoder.fit_transform(x_for_train[:,1])
    x_for_test[:,1]=lab_encoder.fit_transform(x_for_test[:,1])

    #For City Category:
    x_for_train[:,3]=lab_encoder.fit_transform(x_for_train[:,3])
    x_for_test[:,3]=lab_encoder.fit_transform(x_for_test[:,3])

    #--------------------------------------------------------------------------------------------------------

    #Now we gonna do scaling:
    scalar=StandardScaler()
    x_for_train=scalar.fit_transform(x_for_train)

    test_scalar=StandardScaler()
    x_for_test=test_scalar.fit_transform(x_for_test)

    #--------------------------------------------------------------------------------------------------------

    end=time.time()
    print("Time taken for pre-processing for the Gaussian Naive Bayes model is:",end-start,"seconds")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #--------------------------------------------------------------------------------------------------------

    gnb=GaussianNB()
    gnb.fit(x_for_train,y_train_bin)
    pred=gnb.predict(x_for_test)

    #--------------------------------------------------------------------------------------------------------

    end=time.time()
    print("Time taken for execution of the Gaussian Naive Bayes is:",end-start,"seconds")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #--------------------------------------------------------------------------------------------------------

    print("The confusion matrix for the Gaussian Naive Bayes model is as follows:")
    print(confusion_matrix(y_test_bin,pred))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("The classification report for the Gaussian Naive Bayes model is as follows:")
    print(classification_report(y_test_bin,pred))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("Accuracy for the Gaussian Naive Bayes model:",(accuracy_score(y_test_bin,pred))*100,"%")
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#For Linear Regression:

def lin_reg(clean_df):

    for_reg=pd.DataFrame()

    for_reg["Purchase"]=clean_df["Purchase"]
    #For gender, "F" is the baseline i.e. dummy 0
    for_reg["Gender"]=0
    for_reg.loc[(clean_df.Gender=="M"),"Gender"]=1

    #6 for age groups.
    #For Age, 0-17 is baseline (All age dummies 0): 0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55 (signifies 55+)
    for_reg["dum_18-25"]=0
    for_reg["dum_26-35"]=0
    for_reg["dum_36-45"]=0
    for_reg["dum_46-50"]=0
    for_reg["dum_51-55"]=0
    for_reg["dum_55+"]=0

    for_reg.loc[(clean_df.Age=="18-25"),"dum_18-25"]=1
    for_reg.loc[(clean_df.Age=="26-35"),"dum_26-35"]=1
    for_reg.loc[(clean_df.Age=="36-45"),"dum_36-45"]=1
    for_reg.loc[(clean_df.Age=="46-50"),"dum_46-50"]=1
    for_reg.loc[(clean_df.Age=="51-55"),"dum_51-55"]=1
    for_reg.loc[(clean_df.Age=="55"),"dum_55+"]=1

    #So, I am thinking to make 6 dummies for 0,1,4,7,17,20 occupation bins.
    for_reg["dum_0"]=0
    for_reg["dum_1"]=0
    for_reg["dum_4"]=0
    for_reg["dum_7"]=0
    for_reg["dum_17"]=0
    for_reg["dum_20"]=0

    for_reg.loc[(clean_df.Occupation==0),"dum_0"]=1
    for_reg.loc[(clean_df.Occupation==1),"dum_1"]=1
    for_reg.loc[(clean_df.Occupation==4),"dum_4"]=1
    for_reg.loc[(clean_df.Occupation==7),"dum_7"]=1
    for_reg.loc[(clean_df.Occupation==17),"dum_17"]=1
    for_reg.loc[(clean_df.Occupation==20),"dum_20"]=1

    #Gonna take city category C as baseline:
    for_reg["City_A"]=0
    for_reg["City_B"]=0

    for_reg["In_City"]=clean_df["Stay_In_Current_City_Years"]

    for_reg.loc[(clean_df.City_Category=="A"),"City_A"]=1
    for_reg.loc[(clean_df.City_Category=="B"),"City_B"]=1

    for_reg["Marital_Status"]=clean_df["Marital_Status"]

    for_reg["P1_dum"]=clean_df["PC1"]
    for_reg["P5_dum"]=clean_df["PC5"]
    for_reg["P8_dum"]=clean_df["PC8"]

    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(for_reg.head(n=20))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    x=for_reg.iloc[:,1:20].values
    y=for_reg.iloc[:,0].values

    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

    lm=LinearRegression()
    result=lm.fit(x_train,y_train)

    print("Coefficients: ",lm.coef_)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    x2=sm.add_constant(x_train)
    ols=sm.OLS(y_train,x2)
    ols2=ols.fit()
    print(ols2.summary())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    pred=lm.predict(x_test)

    acc=pd.DataFrame(y_test,pred)
    print(acc.head(n=20))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#Reading the data:
df = pd.read_csv("BlackFriday.csv")
print(df.head())
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------

#info will print the number of entries (non-null) and their data type. Describe gives statistical description:
print(df.info())
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(df.describe())
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#--------------------------------------------------------------------------------------------------------

#fillna isn't wokring. Either that or taking just too long:
#Any other methods/suggestions to fill the 'na's?

#df=df.fillna(df.mean()['Product_Category_1':'Product_Category_2'])
#print(df.head())

#--------------------------------------------------------------------------------------------------------

#To see columnwise total number of missing data points:
print(df.isnull().sum())
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

df=df.replace(np.nan,0)
print(df.info())
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

print(df.isnull().sum())
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------

col_names=df.columns.values.tolist()

#0.67% of total data is classified as outliers (For Product Category 1).
#3.5% of total data is classified as outliers (For Product Category 3).
#The percentage is low, so don't think the outliers are wrongful observations, or that they'd drive the mean to a crzy value.
out_idx={}
for col in col_names:
    print("Running for column",col,"Data type:",df[col].dtype)
    if(df[col].dtype=="int64" or df[col].dtype=="float64"):
        out=list(outlier_test(df[col]))
        if len(out[0])>0:
            print("Outliers found in column",col,"Number of outliers:",len(out[0]))
            out_idx[col]=out
    else:
        print("No outliers in column",col)

#--------------------------------------------------------------------------------------------------------

print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("\nNow for clean data frame...\n")
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------

#Taking column names in a list:
col_names=df.columns.values.tolist()

clean_df=pd.DataFrame()

#Columns that won't contribute to machine learning:
to_drop=["User_ID","Product_ID"]
for col in col_names:
    if col in to_drop:
        continue
    else:
        clean_df[col]=df[col]

to_convert=["Product_Category_2","Product_Category_3"]
print("Trying to coerce PC2 and PC3 into int...")
for col in col_names:
    if col in to_convert:
        print("Data type of",col,"is",df[col].dtype)
        clean_df[col]=clean_df[col].astype(int)
        print("Now it is",df[col].dtype)

print("Printing head...")
print(clean_df.head())

#--------------------------------------------------------------------------------------------------------

to_clean=["Stay_In_Current_City_Years"]
col_names1=clean_df.columns.values.tolist()
#Not making this was screwing with the cleaning of '+' cause I was using the column names list from unclean dataframe.
#Cleaning the data for '+' signs. I need to put this out in the comments: Categories: '4' for Years in city and '55' for age are same as 4+ and 55+. May replace those in future

col="Stay_In_Current_City_Years"
clean_df[col]=clean_df[col].apply(lambda x: x.strip("+")).astype(int)

col="Age"
clean_df[col]=clean_df[col].apply(lambda x: x.strip("+"))

#--------------------------------------------------------------------------------------------------------

out_idx={}
for col in col_names1:
    print("Running for column",col,"Data type:",df[col].dtype)
    if(clean_df[col].dtype=="int64" or clean_df[col].dtype=="float64"):
        out=list(outlier_test(clean_df[col]))
        if len(out[0])>0:
            print("Outliers found in column",col,"Number of outliers:",len(out[0]))
            out_idx[col]=out
    else:
        print("No outliers in column",col)

#--------------------------------------------------------------------------------------------------------

#Now visualizations remain
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(clean_df.describe())
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

clean_df["PC1"]=0
clean_df["PC5"]=0
clean_df["PC8"]=0

clean_df.loc[(clean_df.Product_Category_1==1)|(clean_df.Product_Category_2==1)|(clean_df.Product_Category_3==1),"PC1"]=1
clean_df.loc[(clean_df.Product_Category_1==5)|(clean_df.Product_Category_2==5)|(clean_df.Product_Category_3==5),"PC5"]=1
clean_df.loc[(clean_df.Product_Category_1==8)|(clean_df.Product_Category_2==8)|(clean_df.Product_Category_3==8),"PC8"]=1

print(clean_df.head(n=20))
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------

#Testing the max function...:
#print(max(clean_df.Purchase))

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

choice=0

while(choice!=5):
    print("Welcome to the system.")
    print("1). Press 1 for Linear regression model.")
    print("2). Press 2 for Random Forest Regressor.")
    print("3). Press 3 for Random Forest Classifier.")
    print("4). Press 4 for Gaussian Naive Bayes.")
    print("5). Press 5 to exit.")
    choice=input("Your choice: ")
    choice=int(choice)
    if choice==1:
        start=time.time()
        lin_reg(clean_df)
        end=time.time()
        print("Time taken for the pre-rpocessing and execution of Linear regression model is:",end-start,"seconds.")
    elif choice==2:
        forest_regressor(clean_df)
    elif choice==3:
        forest_classifier(clean_df)
    elif choice==4:
        naive_bayes(clean_df)
    elif choice==5:
        print("Goodbye!")
    else:
        print("Invalid choice! Choose again...")
