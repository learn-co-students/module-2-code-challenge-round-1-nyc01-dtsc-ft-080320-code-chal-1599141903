
# Module 2 Assessment

Welcome to your Mod 2 Assessment. You will be tested for your understanding of concepts and ability to solve problems that have been covered in class and in the curriculum.

Use any libraries you want to solve the problems in the assessment.

_Read the instructions carefully_. You will be asked both to write code and respond to a few short answer questions.

**Note on the short answer questions**: For the short answer questions please use your own words. The expectation is that you have not copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, you should do your best to communicate yourself clearly.

The sections of the assessment are:
- Statistical Distributions
- Statistical Tests
- Bayesian Statistics
- Linear Regression and Extensions


```python
# import the necessary libraries
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
## Part 1: Statistical Distributions [Suggested time: 25 minutes]
---

### a. Normal Distributions

Say we have check totals for all checks ever written at a TexMex restaurant. 

The distribution for this population of check totals happens to be normally distributed with a population mean of $\mu = 20$ and population standard deviation of $\sigma = 2$. 

1.a.1) Write a function to compute the z-scores for single checks of amount `check_amt`.


```python
def z_score(check_amt):
    """
    check_amt = the amount for which we want to compute the z-score
    """
    pass
```

1.a.2) I go to the TexMex restaurant and get a check for 24 dollars. 

Use your function to compute your check's z-score, and interpret the result using the empirical rule. 


```python
# your code here 
```


```python
# your answer here
```

1.a.3) Using $\alpha = 0.05$, is my 25 dollar check significantly **greater** than the mean? How do you know this?  

Hint: Here's a link to a [z-table](https://www.math.arizona.edu/~rsims/ma464/standardnormaltable.pdf). 


```python
# your code here 
```


```python
# your answer here 
```

### b. Confidence Intervals and the Central Limit Theorem

1.b.1) Determine the 95% confidence interval around the mean check total for this population. Interpret your result. 


```python
# your code here 
```


```python
# your written answer here
```

1.b.2) Imagine that we didn't know how the population of check totals was distributed. How would **sampling** and the **Central Limit Theorem** allow us to **make inferences on the population mean**, i.e. estimate $\mu, \sigma$ of the population mean?


```python
# Your written answer here
```

---
## Part 2: Statistical Testing [Suggested time: 15 minutes]
---

The TexMex restaurant recently introduced Queso to its menu.

We have random samples of 1000 "No Queso" order check totals and 1000 "Queso" order check totals for orders made by different customers.

In the cell below, we load the sample data for you into the arrays `no_queso` and `queso` for the "no queso" and "queso" order check totals. Then, we create histograms of the distribution of the check amounts for the "no queso" and "queso" samples. 


```python
# Load the sample data 
no_queso = pickle.load(open("data/no_queso.pkl", "rb"))
queso = pickle.load(open("data/queso.pkl", "rb"))

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

### a. Hypotheses and Errors

The restaurant owners want to know if customers who order Queso spend **more or less** than customers who do not order Queso.

2.a.1) Set up the null $H_{0}$ and alternative hypotheses $H_{A}$ for this test.


```python
# Your written answer here
```

2.a.2) What does it mean to make `Type I` and `Type II` errors in this specific context?


```python
# your answer here
```

### b. Sample Testing

2.b.1) Run a statistical test on the two samples. Use a significance level of $\alpha = 0.05$. You can assume the two samples have equal variance. Can you reject the null hypothesis? 

_Hint: Use `scipy.stats`._


```python
# your code here 
```


```python
# your answer here
```

---
## Part 3: Bayesian Statistics [Suggested time: 15 minutes]
---
### a. Bayes' Theorem

Thomas wants to get a new puppy 🐕 🐶 🐩 


<img src="https://media.giphy.com/media/rD8R00QOKwfxC/giphy.gif" />

He can choose to get his new puppy either from the pet store or the pound. The probability of him going to the pet store is $0.2$. 

He can choose to get either a big, medium or small puppy.

If he goes to the pet store, the probability of him getting a small puppy is $0.6$. The probability of him getting a medium puppy is $0.3$, and the probability of him getting a large puppy is $0.1$.

If he goes to the pound, the probability of him getting a small puppy is $0.1$. The probability of him getting a medium puppy is $0.35$, and the probability of him getting a large puppy is $0.55$.

3.a.1) What is the probability of Thomas getting a small puppy? 

3.a.2) Given that he got a large puppy, what is the probability that Thomas went to the pet store?

3.a.3) Given that Thomas got a small puppy, is it more likely that he went to the pet store or to the pound?

3.a.4) For Part 2, what is the prior, posterior and likelihood?


```python
ans1 = None
ans2 = None
ans3 = "answer here"
ans4_prior = "answer here"
ans4_posterior = "answer here"
ans4_likelihood = "answer here"
```

---
## Part 4: Linear Regression and extensions [Suggested Time: 25 min]
---

In this section, you'll be using the Advertising data, and you'll be creating linear models that are more complicated than a simple linear regression. The relevant modules have already been imported at the beginning of this notebook. We'll load and prepare the dataset for you below.


```python
data = pd.read_csv('data/advertising.csv').drop('Unnamed: 0',axis=1)
data.describe()
```


```python
X = data.drop('sales', axis=1)
y = data['sales']
```


```python
# split the data into training and testing set. Do not change the random state please!
X_train , X_test, y_train, y_test = train_test_split(X, y,random_state=2019)
```

### a. Multiple Linear Regression

In the linear regression section of the curriculum, you've analyzed how TV, Radio and Newspaper spendings individually affected the Sales figures. Here, we'll use all three together in a multiple linear regression model!

4.a.1) Create a Correlation Matrix for `X`.


```python
# your code here 
```

4.a.2) Based on this correlation matrix only, would you recommend to use `TV`, `radio` and `newspaper` in the same multiple linear regression model?


```python
# Your written answer here
```

4.a.3) Use StatsModels' `ols`-function to create a multiple linear regression model with `TV`, `radio` and `newspaper` as independent variables and sales as the dependent variable. Use the **training set only** to create the model.

Required output: the model summary of this multiple regression model.


```python
# your code here 
```

4.a.4) Do we have any statistically significant coefficients? If the answer is yes, list them below.


```python
# Your written answer here
```

### b. Polynomial terms

We'd like to add a bit of complexity to the model created in the example above, and we will do it by adding some polynomial terms. Write a function to calculate train and test error for different polynomial degrees. You'll use `scikit-learn`'s `PolynomialFeatures`.

4.b.1) Create the function described above. This function should:
* take `degree` as a parameter that will be used to create polynomial features to be used in a linear regression model
* create a PolynomialFeatures object for each degree and fit a linear regression model using the transformed data
* calculate the mean square error for each level of polynomial
* return the `train_error` and `test_error` 


```python
def polynomial_regression(degree):
    """
    Calculate train and test errorfor a linear regression with polynomial features.
    (Hint: use PolynomialFeatures)
    
    input: Polynomial degree
    output: Mean squared error for train and test set
    """
    # // your code here //
    
    train_error = None
    test_error = None
    return train_error, test_error
```

4.b.2) Try out your new function



```python
polynomial_regression(3)
```

#### Check your answers

MSE for degree 3:
- Train: 0.2423596735839209
- Test: 0.15281375973923944

MSE for degree 4:
- Train: 0.18179109317368244
- Test: 1.9522597174462015

4.b.3) What is the optimal number of degrees for our polynomial features in this model? In general, how does increasing the polynomial degree relate to the Bias/Variance tradeoff?  (Note that this graph shows RMSE and not MSE.)

<img src ="visuals/rsme_poly_2.png" width = "600">

<!---
fig, ax = plt.subplots(figsize=(7, 7))
degree = list(range(1, 10 + 1))
ax.plot(degree, error_train[0:len(degree)], "-", label="Train Error")
ax.plot(degree, error_test[0:len(degree)], "-", label="Test Error")
ax.set_yscale("log")
ax.set_xlabel("Polynomial Feature Degree")
ax.set_ylabel("Root Mean Squared Error")
ax.legend()
ax.set_title("Relationship Between Degree and Error")
fig.tight_layout()
fig.savefig("visuals/rsme_poly.png",
            dpi=150,
            bbox_inches="tight")
--->


```python
# Your answer here
```

### d. In general what methods would you can use to reduce overfitting and underfitting? Provide an example for both and explain how each technique works to reduce the problems of underfitting and overfitting.


```python
# Your answer here
```
