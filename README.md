# Humana-Mays-Healthcare-Case-Competition
### Brief Introduction
- Mays Business School in partnership with Humana presents the fourth annual Humana-Mays Healthcare Analytics Case Competition. This competition offers an opportunity for U.S. masters students to showcase their analytical skills and solve a real-world business problems for Humana utilizing real data. 

- Last year, over 1,300 masters level students representing over 80 major universities in the U.S. registered for the national competition to compete for $52,500 in total prizes. The case competition is open to all accredited educational institutions based in the United States. Full-time and part-time master’s students from accredited Master of Science, Master of Arts, Master of Information Systems, Master of Public Health, Master of Business Administration programs, or other similar master’s programs in business, healthcare, or analytics, are eligible to enter.
### Case Description
Social determinants of health are the conditions in the environments in which people live, learn, work, play, worship and age that affect a wide range of health, functioning and quality-of-life outcomes and risks.  Transportation challenges is one of these determinants.  
- Using the data provided and potentially supplementing with public data, create a model to predict which Medicare members are most likely struggling with Transportation Challenges.
- Propose solutions for overcoming these barrier to accessing care and achieving their best health. 

### The Goal
In the absence of regular, universal screening for SDoH, Humana needs to utilize robust data and advanced data science to understand which of our members are struggling with SDoH. This challenge will focus on Transportation Challenges, so provided with member data that can be supplemented with public data, the goal is to  identify Medicare members most likely experiencing Transportation Challenges and propose viable solutions. 

### Key Components
##### Definitions
- Transportation screening question is coming from the Accountable Health Communities – Health Related Social Needs Screening Tool.
- The question reads: “In the past 12 months,has a lack of reliable transportation kept you from medical appointments, meetings, work or from getting things needed for daily living?” Yes / No
- The date the survey was completed is on the file.

##### Challenging Problem
- **Predictive model** - Since screening all Medicare members is challenging, having a effective predictive model to accurately identify members most likely struggling with Transportation Challenges is valuable. Data is provided and can be supplemented with publically available data.
- **Proposed solutions** – It is likely that members struggling with Transportation Challenges are not homogeneous and hence there are perhaps differentsolutions for different segments of members.

##### Data Included
- Medical claims features
- Pharmacy claims features
- Lab claims features
- Demographic / Consumer data
- Credit data features
- Clinical Condition related features
- CMS Member Data elements
- Other features

##### Group Members
We are a group of three: 
- Bufan Wang
- Zihao Zhao (Me)
- Xiangyu Huang

### Solution Framework
With more than eighty hundred variables in the datasets, we started our analysis with picking up the most relatable variables and deleting irrelevant ones. We divided all the explanatory variables into three groups, binary variables, numerical variables and categorical variables. For the numerical variables, we selected twenty of them which have the highest correlations with the response variable. For the categorical variables, we arbitrarily excluded those unrelated variables and keep the most related ones, which are sex_cd, lang_spoken_cd and rucc_category. Then, we turned them into dummy variables. For the rucc_category, we replaced the ‘Metro’ ones with 0 and ‘Nonmetro’ ones with 1. For the binary variables, we only selected those with reasonable range of transportation issue distribution. As a result, we were left with 42 explanatory variables. After the data pre-processing (mainly fixed missing values and removed abnormal values), we built a binary classfication model (based on ridge regression) to fit our cleaned data, due to the limited computing power, we did not try a resonable number of penality factor (alpha). We ended up with getting a f1-score of 0.85, which earned us a place in top 50 teams.
