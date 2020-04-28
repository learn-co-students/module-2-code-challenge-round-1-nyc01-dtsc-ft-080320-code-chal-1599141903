
# Module 2 Code Challenge

Welcome to your Module 2 Code Challenge. This code challenge is designed to test your understanding of the Module 2 material. It covers:

- Statistical Distributions
- Statistical Tests
- Bayesian Statistics
- Linear Regression

_Read the instructions carefully_. You will be asked both to write code and respond to a few short answer questions.

### Note on the short answer questions: 

For the short answer questions _please use your own words_. The expectation is that you have **not** copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, you should do your best to communicate yourself clearly.


```python
# Run this cell without changes to import the necessary libraries

# Use any additional libraries you like to complete this assessment 

import itertools
import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import pickle

from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
```

---
## Part 1: Statistical Distributions [Suggested time: 20 minutes]
---

### Normal Distributions

Let's consider check totals at a TexMex restaurant. We know that the population distribution of check totals is normally distributed with a mean of $\mu$ = \\$20 and standard deviation of $\sigma$ = \\$3. 

### 1.1) Compute the z-score for a \\$26 check. 


```python
# Code here 
```

### 1.2) Approximately what percentage of all checks are less than \\$26? Explain how you came to your answer.

You can answer this using the empirical rule or this [z-table](https://www.math.arizona.edu/~rsims/ma464/standardnormaltable.pdf).


```python
"""
Written answer here
"""
```

### Confidence Intervals

One night, a waiter gets 15 checks with a mean of \\$19 and a standard deviation of \\$3.

### 1.3) Calculate the 95% confidence interval around the mean for this waitor's checks. Interpret the result. 


```python
# Code here 
```


```python
"""
Written answer here
"""
```

---
## Part 2: Statistical Testing [Suggested time: 20 minutes]
---

The TexMex restaurant recently introduced queso to its menu.

We have random samples of 1000 "no queso" order check totals and 1000 "queso" order check totals for orders made by different customers.

In the cell below, we load the sample data for you into the arrays `no_queso` and `queso` for the "no queso" and "queso" order check totals, respectively. Then, we create histograms of the distribution of the check amounts for the "no queso" and "queso" samples. 


```python
# Run this cell without changes

# Load the sample data 
no_queso = pickle.load(open('data/no_queso.pkl', 'rb'))
queso = pickle.load(open('data/queso.pkl', 'rb'))

# Plot histograms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.set_title('Sample of Non-Queso Check Totals')
ax1.set_xlabel('Amount')
ax1.set_ylabel('Frequency')
ax1.hist(no_queso, bins=20)

ax2.set_title('Sample of Queso Check Totals')
ax2.set_xlabel('Amount')
ax2.set_ylabel('Frequency')
ax2.hist(queso, bins=20)
plt.show()
```

### Hypotheses and Errors

The restaurant owners want to know if customers who order queso spend **more or less** than customers who do not order queso.

### 2.1) Set up the null $H_{0}$ and alternative hypotheses $H_{A}$ for this test.


```python
"""
Written answer here
"""
```

### 2.2) What does it mean to make a `Type I` error or a `Type II` error in this specific context?


```python
"""
Written answer here
"""
```

### Sample Testing

### 2.3) Run a statistical test on the two samples. Can you reject the null hypothesis? 

Use a significance level of $\alpha = 0.05$. You can assume the two samples have equal variance.

You can use `scipy.stats` to find the answer if you like.  It has already been imported as `stats` and the statistical testing documentation can be found [here](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests).


```python
# Code here 
```


```python
"""
Written answer here
"""
```

---
## Part 3: Bayesian Statistics [Suggested time: 15 minutes]
---
### Bayes' Theorem

A medical test is designed to diagnose a certain disease. The test has a false positive rate of 10%, meaning that 10% of people without the disease will get a positive test result. The test has a false negative rate of 2%, meaning that 2% of people with the disease will get a negative result. Only 1% of the population has this disease.

### 3.1) If a patient receives a positive test result, what is the probability that they actually have the disease? Show how you arrive at your answer.


```python
# Code here to find the probability
```

---
## Part 4: Linear Regression [Suggested Time: 20 min]
---

In this section, you'll be using the Advertising data to run regression models. In this dataset, each row represents a different product, and we have a sample of 200 products from a larger population of products. We have three features - `TV`, `radio`, and `newspaper` - that describe how many thousands of advertising dollars were spent promoting the product. The target, `sales`, describes how many millions of dollars in sales the product had.

The relevant modules have already been imported at the beginning of this notebook. We'll load and prepare the dataset for you below.


```python
# Run this cell without changes

data = pd.read_csv('data/advertising.csv').drop('Unnamed: 0', axis=1)
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>radio</th>
      <th>newspaper</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>147.042500</td>
      <td>23.264000</td>
      <td>30.554000</td>
      <td>14.022500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>85.854236</td>
      <td>14.846809</td>
      <td>21.778621</td>
      <td>5.217457</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.700000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>74.375000</td>
      <td>9.975000</td>
      <td>12.750000</td>
      <td>10.375000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>149.750000</td>
      <td>22.900000</td>
      <td>25.750000</td>
      <td>12.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>218.825000</td>
      <td>36.525000</td>
      <td>45.100000</td>
      <td>17.400000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>296.400000</td>
      <td>49.600000</td>
      <td>114.000000</td>
      <td>27.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Run this cell without changes

X = data.drop('sales', axis=1)
y = data['sales']
```

### Simple Linear Regression

### 4.1) Use StatsModels' `ols`-function to run a linear regression model using `TV` to predict `sales`. 

**Required output:** the summary of this regression model.


```python
# Code here 
```

### 4.2) Do these results imply that _all_ products that receive more TV advertising have greater sales? Explain how you determined this based on the model output. 


```python
"""
Written answer here
"""
```

### Multiple Linear Regression

### 4.3) Compute a correlation matrix for `X`. Does this suggest that any of the features are strongly collinear? 


```python
# Code here 
```


```python
"""
Written answer here
"""
```

### 4.4) Use StatsModels' `ols`-function to run a multiple linear regression model with `TV`, `radio`, and `newspaper` as independent variables and `sales` as the dependent variable. 

**Required output:** the summary of this regression model.


```python
# Code here 
```

### 4.5) Does this model do a better job of predicting sales than the previous model using only the `TV` feature? Explain how you determined this based on the model output. 


```python
"""
Written answer here
"""
```
