# %% [markdown]
"""
# Machine Learning - Practical 1 - Linear Regression

Names: {YOUR NAMES}  
Summer Term 2024   
"""

# %% [markdown]
"""
This notebook provides you with the assignments and the overall code structure you need to complete the assignment. There are also questions that you need to answer in text form. Please use full sentences and reasonably correct spelling/grammar.

Regarding submission & grading:

- Work in groups of three and hand in your solution as a group.

- Solutions need to be uploaded to StudIP until the submission date indicated in the course plan. Please upload a copy of this notebook and a PDF version of it after you ran it.

- Solutions need to be presented to tutors in tutorial. Presentation dates are listed in the course plan. Every group member needs to be able to explain everything.

- You have to solve N-1 practicals to get admission to the exam.

- For plots you create yourself, all axes must be labeled. 

- Do not change the function interfaces.
"""

# %% [markdown]
"""
## Imports

Jupyter Notebook provides the possibility of using libraries, functions and variables globally. This means, once you import the libraries, functions, etc. you won't have to import them again in the next cell. However, if for any reason you end the session (crash, timeout, etc.), then you'll have to run this cell to have your libraries imported again. So, let's go ahead and import whatever we need in this homework assignment.
"""

# %%
# %matplotlib inline

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
"""
## The  dataset
"""

# %% [markdown]
"""
The dataset consists of over 20.000 materials and lists their physical features. From these features, we want to learn how to predict the critical temperature, i.e. the temperature we need to cool the material to so it becomes superconductive. First load and familiarize yourself with the data set a bit.
"""

# %%
data = pd.read_csv('Practical_1/superconduct_train.csv')
print(data.shape)

# %%
data.head()

# %% [markdown]
"""
Because the dataset is rather large, we prepare a small subset of the data as training set, and another subset as test set. To make the computations reproducible, we set the random seed. This makes the train and test splits same even if you re-run the notebook. Keeping the splits same is important for the fair models comparison.
"""

# %%
target_clm = 'critical_temp'  # the critical temperature is our target variable
n_trainset = 200  # size of the training set
n_testset = 500  # size of the test set

# %%
# set random seed to make sure every test set is the same
np.random.seed(seed=1)

idx = np.arange(data.shape[0])
idx_shuffled = np.random.permutation(idx)  # shuffle indices to split into training and test set

test_idx = idx_shuffled[:n_testset]
train_idx = idx_shuffled[n_testset:n_testset+n_trainset]
train_full_idx = idx_shuffled[n_testset:]

X_test = data.loc[test_idx, data.columns != target_clm].values
y_test = data.loc[test_idx, data.columns == target_clm].values
print('Test set shapes (X and y)', X_test.shape, y_test.shape)

X_train = data.loc[train_idx, data.columns != target_clm].values
y_train = data.loc[train_idx, data.columns == target_clm].values
print('Small training set shapes (X and y):', X_train.shape, y_train.shape)

X_train_full = data.loc[train_full_idx, data.columns != target_clm].values
y_train_full = data.loc[train_full_idx, data.columns == target_clm].values
print('Full training set shapes (X and y):', X_train_full.shape, y_train_full.shape)

# %% [markdown]
"""
## Task 1: Plot the dataset

To explore the dataset, use `X_train_full` and `y_train_full` for two descriptive plots:

* **Histogram** of the target variable. Use `plt.hist`.

* **Scatter plots** relating the target variable to one of the feature values. For this you will need 81 scatter plots. Arrange them in one big figure with 9x9 subplots. Use `plt.scatter`. You may need to adjust the marker size and the alpha blending value. 

Furthermore, we need to normalize the data, such that each feature has a mean of zero mean and a variance of one. Implement a function `normalize` which normalizes the data. Print the means and standard variation of the first five features before and after.
"""

# %%
# Histogram of the target variable


# %%
# Scatter plots of the target variable vs. features

GRID_ROWS = 9
GRID_COLS = 9
DOT_SIZE = 1
FIG_SIZE = 10
TITLE_FONT_SIZE = 5
SCATTER_TITLES = [col.replace('_', '\n') for col in data.columns]

def plot_data(X, Y, titles=SCATTER_TITLES):
    """
    Scatter each of the GRID_ROWS×GRID_COLS columns of X against Y.

    Parameters
    ----------
    X : array-like of shape (n_samples, GRID_ROWS*GRID_COLS)
    Y : array-like of shape (n_samples,)
    titles : list of str, length GRID_ROWS*GRID_COLS
        The title for each subplot (underscores will be replaced with line breaks).
    """
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(FIG_SIZE, FIG_SIZE))
    
    for idx, ax in enumerate(axes.flat):
        ax.scatter(X[:, idx], Y, s=DOT_SIZE)
        ax.set_title(titles[idx].replace('_', '\n'),
                     fontsize=TITLE_FONT_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    return fig, axes

# %%
_,_ = plot_data(X_train, y_train)


# %%
# Normalize

def normalize_one(data_column):
   normalized = np.zeros(np.shape(data_column))
   mean = np.mean(data_column)
   std = np.std(data_column)
   for i in range(len(data_column)):
       normalized[i] = (data_column[i]-mean)/std

   return normalized

def normalize(data):
   normalized = np.zeros(np.shape(data))
   cols = np.shape(data)[1]
   for c in range(cols):
     normalized_cols = normalize_one(data[:,c])
     normalized[:, c] = normalized_cols

   return normalized

# %%

X_train_norm = normalize(X_train)
y_train_norm = normalize(y_train)

# %%

_,_ = plot_data(X_train_norm, y_train_norm)



# %% [markdown]
"""
Which material properties may be useful for predicting superconductivity? What other observations can you make?
"""

# %% [markdown]
"""
 YOUR ANSWER HERE
"""

# %% [markdown]
"""
## Task 2:  Implement your own OLS estimator

We want to use linear regression to predict the critical temperature. Implement the ordinary least squares estimator without regularization 'by hand':

$w = (X^TX)^{-1}X^Ty$

To make life a bit easier, we provide a function that can be used to plot regression results. In addition it computes the mean squared error and the squared correlation between the true and predicted values. 
"""

# %%
def plot_regression_results(y_test, y_pred, weights):
    '''Produces three plots to analyze the results of linear regression:
        -True vs predicted
        -Raw residual histogram
        -Weight histogram

    Inputs:
        y_test: (n_observations,) numpy array with true values
        y_pred: (n_observations,) numpy array with predicted values
        weights: (n_weights) numpy array with regression weights'''

    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('r^2: ', r2_score(y_test, y_pred))

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    # predicted vs true
    ax[0].scatter(y_test, y_pred, s=2)
    ax[0].set_title('True vs. Predicted')
    ax[0].set_xlabel('True %s' % (target_clm))
    ax[0].set_ylabel('Predicted %s' % (target_clm))

    # residuals
    error = np.squeeze(np.array(y_test)) - np.squeeze(np.array(y_pred))
    ax[1].hist(np.array(error), bins=30)
    ax[1].set_title('Raw residuals')
    ax[1].set_xlabel('(true-predicted)')

    # weight histogram
    ax[2].hist(weights, bins=30)
    ax[2].set_title('weight histogram')

    plt.tight_layout()

# %% [markdown]
"""
As an example, we here show you how to use this function with random data. 
"""

# %%
# weights is a vector of length 82: the first value is the intercept (beta0), then 81 coefficients
weights = np.random.randn(82)

# Model predictions on the test set
y_pred_testing = np.random.randn(y_test.size) * np.max(y_test)

plot_regression_results(y_test, y_pred_testing, weights)

# %% [markdown]
"""
Implement OLS linear regression yourself. Use `X_train` and `y_train` for estimating the weights and compute the MSE and $r^2$ from `X_test`. When you call our plotting function with the regression result, you should get mean squared error of 707.8.
"""

# %%
def ols_regression(X_test, X_train, y_train):
    '''Computes OLS weights for linear regression without regularization on the training set and
       returns weights and testset predictions.

       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set

       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set

       Note:
         Both the training and the test set need to be appended manually by a columns of 1s to add
         an offset term to the linear regression model.
    '''

    # Add offset term
    n_train, p = X_train.shape
    n_test  = X_test.shape[0]

    X_train_aug = np.hstack([np.ones((n_train, 1)), X_train]) # (n_train, p+1)
    X_test_aug  = np.hstack([np.ones((n_test,  1)), X_test]) # (n_test,  p+1)

    # Closed-form OLS $w = (X^T X)^{-1} X^T y$
    XtX = X_train_aug.T @ X_train_aug
    Xty = X_train_aug.T @ y_train.reshape(-1, 1)

    weights = np.linalg.inv(XtX) @ Xty
    weights = weights.ravel() # flatten into (p+1,)

    # Predict on test set
    y_pred = X_test_aug @ weights # (n_test,)

    return weights, y_pred

# %%
# Plots of the results
weights, y_pred = ols_regression(X_test, X_train, y_train)
plot_regression_results(y_test, y_pred, weights)


# %% [markdown]
"""
What do you observe? Is the linear regression model good?
"""

# %% [markdown]
"""
YOUR ANSWER HERE
"""

# %% [markdown]
"""
## Task 3: Compare your implementation to sklearn

Now, familiarize yourself with the sklearn library. In the section on linear models:

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

you will find `sklearn.linear_model.LinearRegression`, the `sklearn` implementation of the OLS estimator. Use this sklearn class to implement OLS linear regression. Again obtain estimates of the weights on `X_train` and `y_train` and compute the MSE and $r^2$ on `X_test`.

"""

# %%
def sklearn_regression(X_test, X_train, y_train):
    '''Computes OLS weights for linear regression without regularization using the sklearn library on the training set and
       returns weights and testset predictions.

       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set

       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set

       Note:
         The sklearn library automatically takes care of adding a column for the offset.
    '''

    # Learn weights
    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # Test‐Set predictions
    y_pred = model.predict(X_test)

    # Pull out a flat coef array and prepend the intercept
    coefs   = model.coef_.ravel()
    weights = np.insert(coefs, 0, model.intercept_)

    return weights, y_pred

# %%
weights, y_pred = sklearn_regression(X_test, X_train, y_train)
plot_regression_results(y_test, y_pred, weights)

# %% [markdown]
"""
If you implemented everything correctly, the MSE is again 707.8.
"""

# %% [markdown]
"""
Fit the model using the larger training set, `X_train_full` and `y_train_full`, and again evaluate on `X_test`.
"""

# %%
weights, y_pred = sklearn_regression(X_test, X_train_full, y_train_full)
plot_regression_results(y_test, y_pred, weights)

# %% [markdown]
"""
 How does test set performance change? What else changes?
"""

# %% [markdown]
"""
YOU ANSWER HERE
"""

# %% [markdown]
"""
## Task 4: Regularization with ridge regression

We will now explore how a penalty term on the weights can improve the prediction quality for finite data sets. Implement the analytical solution of ridge regression 

$w = (X^TX + \alpha I_D)^{-1}X^Ty$


as a function that can take different values of $\alpha$, the regularization strength, as an input. In the lecture, this parameter was called $\lambda$, but this is a reserved keyword in Python.
"""

# %%
def ridge_regression(X_test, X_train, y_train, alpha):
    '''Computes OLS weights for regularized linear regression with regularization strength alpha
       on the training set and returns weights and testset predictions.

       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         alpha: scalar, regularization strength

       Outputs:
         weights: The weight vector for the regression model including the offset
         y_pred: The predictions on the TEST set

       Note:
         Both the training and the test set need to be appended manually by a columns of 1s to add
         an offset term to the linear regression model.
    '''

    n_train, p = X_train.shape
    n_test = X_test.shape[0]

    X_train_aug = np.hstack([np.ones((n_train, 1)), X_train])  # (n_train, p+1)
    X_test_aug = np.hstack([np.ones((n_test, 1)), X_test])     # (n_test, p+1)

    # Closed-form Ridge solution $w = (X^T X + alpha * I)^(-1) X^T y$
    XtX = X_train_aug.T @ X_train_aug  # (p+1, p+1)
    I = np.eye(XtX.shape[0])           # Identity (p+1, p+1)
    
    # Do not regularze the intercept term (bias):
    I[0, 0] = 0   # no penalty on bias

    Xty = X_train_aug.T @ y_train.reshape(-1, 1) # (p+1, 1)

    weights = np.linalg.inv(XtX + alpha * I) @ Xty # (p+1, 1)
    weights = weights.ravel() # flatten to (p+1,)

    # Predict on test set
    y_pred = X_test_aug @ weights # (n_test,)

    return weights, y_pred

# %% [markdown]
"""
Run the ridge regression on `X_train` with an alpha value of 10 and plot the obtained weights.
"""

# %%
# Run ridge regression with alpha=10
weights, y_pred = ridge_regression(X_test, X_train, y_train, alpha=10)

# Plot regression results
plot_regression_results(y_test, y_pred, weights)


# %% [markdown]
"""
Now test a range of log-spaced $\alpha\text{s}$ (~10-20), which cover several orders of magnitude, e.g. from 10^-7 to 10^7. 

* For each $\alpha$, you will get one model with one set of weights. 
* For each model, compute the error on the test set. 

Store both the errors and weights of all models for later use. You can use the function `mean_squared_error` from sklearn (imported above) to compute the MSE.

"""

# %%
alphas = np.logspace(-7, 7, 20)

n_alphas = alphas.size
n_features = X_train.shape[1] + 1  # +1 for the intercept term

all_weights = np.zeros((n_alphas, n_features))
mse_values  = np.zeros(n_alphas)

# Explore logspace of alphas
for i, alpha in enumerate(alphas):
    w, y_pred = ridge_regression(X_test, X_train, y_train, alpha)
    all_weights[i, :] = w
    mse_values[i]    = mean_squared_error(y_test, y_pred)


# %% [markdown]
"""
Make a single plot that shows for each coefficient how it changes with $\alpha$, i.e. one line per coefficient. Also think about which scale is appropriate for your $\alpha$-axis. You can set this using `plt.xscale(...)`.
"""

# %%
# Plot of coefficients vs. alphas
plt.figure(figsize=(8, 6))
for j in range(n_features):
    plt.plot(alphas, all_weights[:, j])
plt.xscale('log')
plt.xlabel('Regularization strength (α)')
plt.ylabel('Coefficient value')
plt.title('Ridge coefficients as a function of α')
plt.tight_layout()
plt.show()


# %% [markdown]
"""
Why are the values of the weights largest on the left? Do they all change monotonically? 
"""

# %% [markdown]
"""
YOUR ANSWER HERE
"""

# %% [markdown]
"""
Plot how the performance (i.e. the error) changes as a function of $\alpha$. As a sanity check, the MSE value for very small $\alpha$ should be close to the test-set MSE of the unregularized solution, i.e. 708.
"""

# %%
# Plot of MSE  vs. alphas
mse_vals  = np.zeros_like(alphas)

# Compute MSE at each α
for i, α in enumerate(alphas):
    _, y_pred = ridge_regression(X_test, X_train, y_train, α)
    mse_vals[i] = mean_squared_error(y_test, y_pred)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alphas, mse_vals)
plt.xscale('log')
plt.xlabel('Regularization strength (α)')
plt.ylabel('Test-set MSE')
plt.title('Ridge Regression: Test MSE vs. α')
plt.tight_layout()
plt.show()

# %% [markdown]
"""
Which value of $\alpha$ gives the minimum MSE? Is it better than the unregularized model? Why should the curve reach ~700 on the left?
"""

# %% [markdown]
"""
YOUR ANSWER HERE
"""

# %% [markdown]
"""
Now implement the same model using sklearn. Use the `linear_model.Ridge` object to do so.

"""

# %%
def ridge_regression_sklearn(X_test, X_train, y_train, alpha):
    '''Computes OLS weights for regularized linear regression with regularization strength alpha using the sklearn
       library on the training set and returns weights and testset predictions.

       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         alpha: scalar, regularization strength

       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set

       Note:
         The sklearn library automatically takes care of adding a column for the offset.
    '''

    model=linear_model.Ridge(alpha=alpha)#erzeugt Linear model object 
    model.fit(X_train, y_train)#erstellen des models mit den trainingsdaten
    y_pred=model.predict(X_test)#so erhalten wir die pred. values
    weights=model.coef_#so erhalten wir die koeffizienten (=weights)


    return weights, y_pred

# %% [markdown]
"""
This time, only plot how the performance changes as a function of $\alpha$. 
"""

# %%
# Plot of MSE  vs. alphas
mse_vals  = np.zeros_like(alphas)
# Compute MSE at each α
for i, α in enumerate(alphas):
    _, y_pred = ridge_regression_sklearn(X_test, X_train, y_train, α)
    mse_vals[i] = mean_squared_error(y_test, y_pred)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alphas, mse_vals)
plt.xscale('log')
plt.xlabel('Regularization strength (α)')
plt.ylabel('Test-set MSE')
plt.title('Ridge Regression: Test MSE vs. α')
plt.tight_layout()
plt.show()

# %% [markdown]
"""
Note: Don't worry if the curve is not exactly identical to the one you got above. The loss function we wrote down in the lecture  has $\alpha$ defined a bit differently compared to sklearn. However, qualitatively it should look the same.
"""

# %% [markdown]
"""
## Task 5: Cross-validation

Until now, we always estimated the error on the test set directly. However, we typically do not want to tune hyperparameters of our inference algorithms like $\alpha$ on the test set, as this may lead to overfitting. Therefore, we tune them on the training set using cross-validation. As discussed in the lecture, the training data is here split in `n_folds`-ways, where each of the folds serves as a held-out dataset in turn and the model is always trained on the remaining data. Implement a function that performs cross-validation for the ridge regression parameter $\alpha$. You can reuse functions written above.
"""

# %%
def ridgeCV(X, y, n_folds, alphas):
    '''Runs a n_fold-crossvalidation over the ridge regression parameter alpha.
       The function should train the linear regression model for each fold on all values of alpha.

      Inputs:
        X: (n_obs, n_features) numpy array - predictor
        y: (n_obs,) numpy array - target
        n_folds: integer - number of CV folds
        alphas: (n_parameters,) - regularization strength parameters to CV over

      Outputs:
        cv_results_mse: (n_folds, len(alphas)) numpy array, MSE for each cross-validation fold

      Note:
        Fix the seed for reproducibility.
    '''

    cv_results_mse = np.zeros((n_folds, len(alphas)))
    np.random.seed(seed=2)

    n = X.shape[0]
    cv_results_mse = np.zeros((n_folds, len(alphas)))

    # ensure reproducibility
    np.random.seed(2)
    idx = np.random.permutation(n)
    # split into n_folds roughly equal parts
    folds = np.array_split(idx, n_folds)

    for i in range(n_folds):
        # held‐out (validation) indices
        val_idx   = folds[i]
        # all other indices for training
        train_idx = np.hstack([folds[j] for j in range(n_folds) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        # for each alpha, fit and compute MSE on the held‐out fold
        for j, alpha in enumerate(alphas):
            _, y_pred_val = ridge_regression(X_val, X_train, y_train, alpha)
            cv_results_mse[i, j] = mean_squared_error(y_val, y_pred_val)

    return cv_results_mse

# %% [markdown]
"""
Now we run 10-fold cross-validation using the training data of a range of $\alpha$s.
"""

# %%
alphas = np.logspace(-7, 7, 100)
mse_cv = ridgeCV(X_train, y_train, n_folds=10, alphas=alphas)

# %% [markdown]
"""
We plot the MSE trace for each fold separately:
"""

# %%
plt.plot(alphas, mse_cv.T, '.-')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()

# %% [markdown]
"""
We also plot the average across folds:
"""

# %%
plt.figure(figsize=(6, 4))
plt.plot(alphas, np.mean(mse_cv, axis=0), '.-')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()

# %% [markdown]
"""
What is the optimal $\alpha$? Is it similar to the one found on the test set? Do the cross-validation MSE and the test-set MSE match well or differ strongly?
"""

# %% [markdown]
"""
YOUR ANSWER HERE
"""

# %% [markdown]
"""
We will now run cross-validation on the full training data. This will take a moment, depending on the speed of your computer. Afterwards, we will again plot the mean CV curves for the full data set (blue) and the small data set (orange).
"""

# %%
alphas = np.logspace(-7, 7, 100)
mse_cv_full = ridgeCV(X_train_full, y_train_full, n_folds=10, alphas=alphas)

# %%
plt.figure(figsize=(6, 4))
plt.plot(alphas, np.mean(mse_cv_full, axis=0), '.-')
plt.plot(alphas, np.mean(mse_cv, axis=0), '.-')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()

# %% [markdown]
"""
We zoom in on the blue curve to the very left:
"""

# %%
plt.figure(figsize=(6, 4))
plt.plot(alphas, np.mean(mse_cv_full, axis=0), '.-')
plt.xscale('log')
minValue = np.min(np.mean(mse_cv_full, axis=0))
plt.ylim([minValue-.01, minValue+.02])
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()

# %% [markdown]
"""
Why does the CV curve on the full data set look so different? What is the optimal value of $\alpha$ and why is it so much smaller than on the small training set?
"""

# %% [markdown]
"""
YOUR ANSWER HERE
"""

