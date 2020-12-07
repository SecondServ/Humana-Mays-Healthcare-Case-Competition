import pandas as pd
import numpy as np
import  statsmodels.api as sm
from sklearn.linear_model import Ridge,Lasso,LogisticRegression
from sklearn.model_selection import train_test_split

training = pd.read_csv('2020_Competition_Training.csv')

# Collect binary varibales
training_ind = training.loc[:,training.isin([0,1]).all()]
# Find correlations between binary variables and transportation issues
training_ind_corr  = training_ind.corr().iloc[:,0].reset_index()
training_ind_corr = training_ind_corr.iloc[1:, :].dropna()
# Get absolute values of correlations
training_ind_corr['transportation_issues'] = abs(training_ind_corr['transportation_issues'])
# Sort absolute values of correlations
training_ind_corr = training_ind_corr.sort_values(by = 'transportation_issues',ascending = False)
# Select top 20 binary variables with highest absolute values of correlations
training_ind_corr_top20 = training_ind_corr.head(20).set_index('index')
# Locate them in dataset
training_ind_top20 = training_ind.loc[:, [i for i in training_ind_corr_top20.index]]


# Label categorical variables
categorical_bolean = (training.dtypes == "object").values
# Select relevant categorical variables
training_categorical = training.loc[:,categorical_bolean]
training_categorical = training_categorical.iloc[:,[2,3,-4]]
# Turn categorical variable 'rucc_category' to dummy
training_categorical['rucc_category'] = training_categorical['rucc_category'].map({'1-Metro':0,'2-Metro':0,'3-Metro':0,'6-Nonmetro':1,'4-Nonmetro':1,
                                          '7-Nonmetro':1,'8-Nonmetro':1,'9-Nonmetro':1,'5-Nonmetro':1})
# Turn categorical variables to dummy variables                                       
training_dummies = pd.get_dummies(training_categorical,drop_first = True)



# Locate numerical variables
training_numerical = training.loc[:,~categorical_bolean]
# Remove dummy-like variables
training_numerical = training_numerical.loc[:,~training_numerical.isin([0,1]).all()]
# Concatenate numerical columns with transportation issues column
training_numerical = pd.concat([training['transportation_issues'],training_numerical],axis = 1)
# Compute correlations
training_numerical_corr = training_numerical.corr()['transportation_issues']
training_numerical_corr = training_numerical_corr.reset_index()
# Get top 20 highest absolute values of correlations
training_numerical_corr_top20 = training_numerical_corr.set_index('index').abs().sort_values(by = 'transportation_issues',ascending = False).head(21)
# Locate them in dataset
training_numerical_corr_top20 = training_numerical.loc[:,[i for i in training_numerical_corr_top20.index]]
training_numerical_corr_top20 = training_numerical_corr_top20.drop('transportation_issues',axis = 1)



'''Combine all three datasets to get a structured dataset including 20 binary variables, 20 numerical variables, 
3 categorical variables and transportation issue column as dependent variable'''
training_semifinal = pd.concat([training.transportation_issues,training_ind_top20,training_dummies,training_numerical_corr_top20],axis = 1)
# Drop rows with missing values
training_final = training_semifinal.dropna()
# add constant
training_final['intercept'] = 1.0
# Make the 43 selected variables to be independent variables and tranportation issues to be dependent vars
X = training_final.iloc[:,1:]
y = training_final.iloc[:,0]
# Train the model
logit = sm.Logit(y,X)
result = logit.fit()
# Get sorted probabilities of having tranportation issues
score = result.predict(X)
score.sort_values().reset_index()



# Read test data
test_data = pd.read_csv('2020_Competition_Holdout .csv')
# Preprocess using the same logic to get structured dataset as input
test_data_ind = test_data.loc[:,test_data.isin([0,1]).all()]
test_data_ind = test_data_ind.loc[:, [i for i in training_ind_corr_top20.index[:]]]
test_data_ind.fillna(test_data_ind.mode(),inplace = True)
categorical_bolean_test = (test_data.dtypes == "object").values
test_categorical = test_data.loc[:,categorical_bolean_test]
test_categorical = test_categorical.iloc[:,[2,3,-4]]
test_categorical['rucc_category'] = test_categorical['rucc_category'].map({'1-Metro':0,'2-Metro':0,'3-Metro':0,'6-Nonmetro':1,'4-Nonmetro':1,
                                          '7-Nonmetro':1,'8-Nonmetro':1,'9-Nonmetro':1,'5-Nonmetro':1})

test_dummies = pd.get_dummies(test_categorical, drop_first = True)
test_numerical = test_data.loc[:,[i for i in training_numerical_corr_top20.columns]]
test_numerical.fillna(test_numerical.mean(),inplace = True)
test_final = pd.concat([test_data_ind,test_dummies,test_numerical],axis = 1)
# Add constant
X_test = test_final.assign(intercept = [1.0 for i in test_final.index ])



# Fit
score2 = result.predict(X_test)
score2 = pd.DataFrame({'Score': score2})
# Sort scores
semifinal_score = score2.sort_values(by = 'Score',ascending = False).reset_index()
# Match to Customer IDs
final_score = semifinal_score.assign(Rank = [i for i in range(1,17682)])
# Get regression result
result.summary()



'''Then, we went over this result and removed those variables which has a p-value greater 
than 0.05 and re-run the logistics regression on the remaining variables to get a final output.'''

# Select low-pvalue variables
pvLow = ['ccsp_239_ind','cms_low_income_ind','cms_disabled_ind','cms_dual_eligible_ind',
        'cmsd2_men_men_substance_ind','bh_cdto_ind','bh_bipr_ind','cmsd2_sns_general_ind',
        'sex_cd_M','est_age','cons_n65p_y','med_er_visit_ct_pmpm','total_ambulance_visit_ct_pmpm',
        'cms_ma_risk_score_nbr','intercept']
# Locate these variables in training dataset
training_final_lowP = training_final.loc[:,pvLow]
# This time independent varibles are the 15 selected low-pvalue varibales
X = training_final_lowP
y = training_final.iloc[:,0]
# Run again to train the model
logit = sm.Logit(y,X)
result2 = logit.fit()
result2.summary()
# Obtain test data
test_final_lowP = test_final.loc[:,pvLow[0:-1]]
# Add constant
X_test = test_final_lowP.assign(intercept = [1.0 for i in test_final.index ])
# Predict porbabities
result2.predict(X_test)
# Add rank and customer ID columns
Finalsocre = pd.DataFrame({'Score':result2.predict(X_test).sort_values(ascending =False)})
Finalsocre = Finalsocre.reset_index().rename({'index':'ID'}).assign(Rank = [i for i in range(1,17682)])
Finalsocre = Finalsocre.set_index('index')