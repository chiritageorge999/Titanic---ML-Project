  *Titanic Project Walk Through*

  Overview
  1. Understand the data
  2. Data Cleaning and exploration
  3. Feature Engineering
  4. Data preprocessing for the model
  5. Basic Model Building
  6. Model Tuning
  7. Ensemble Model Building
  8. Results

 Importing essential libraries for data analysis and visualization
```
import numpy as np                  # Linear algebra
import pandas as pd                 # Data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns               # Statistical data visualization
import matplotlib.pyplot as plt     # Plotting graphs and charts
```
Importing the data for analysis - using exclusively the Training set.

```
training = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

training['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([training,test])

%matplotlib inline
all_data.columns
```
Project Planning
When starting a new project, I like to outline the steps I plan on taking.
Below is a rough outline.

* Understand the data .info() .describe()
* Histograms and boxplots
* Value counts
* Missing data
* Correlation between the metrics
* Explore interesting themes
   * Will the wealthy survive?
   * By location
   * Age scatterplot with ticket price
   * Young and wealthy Variable?
   * Total spent?
* Feature engineering
* Preprocess the data
   * Use labels for training and testing
* Model baseline
* Model comparison

  Data Exploration
  
1. For numeric data 
  * Histograms to understand distributions
  * Corrplot
  * Pivot table comparing survival rate across numeric variables

2. For categorical data
  * Bar charts to understand the balance of classes
  * Pivot tables to understand the relationship with survival

```
#quick look at our data types & null counts 
training.info()
```
```
# to better understand the numeric data, I'll use the .describe() method.
# This gives me an understanding of the central tendencies of the data 
training.describe()
```
```
#quick way to separate numeric columns
training.describe().columns
```
```
# look at numeric and categorical values separately 
df_num = training[['Age','SibSp','Parch','Fare']]
df_cat = training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
```
```
#distributions for all numeric variables 
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()
```

```
# compare survival rate across Age, SibSp, Parch, and Fare 
pd.pivot_table(training, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
```
```
for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    plt.show()\
```
Cabin and ticket graphs are very messy. This is an area where I will do some feature engineering.

```
# Comparing survival and each of these categorical variables 
print(pd.pivot_table(training, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(training, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))
print()
print(pd.pivot_table(training, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))
```












