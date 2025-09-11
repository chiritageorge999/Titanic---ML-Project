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
