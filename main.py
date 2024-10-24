import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# Load and split data
train_data = pd.read_csv("InjectionMolding_Train.csv")
X_train = train_data.drop(['mass'], axis=1)
Y_train = train_data['mass']

test_data = pd.read_csv("InjectionMolding_Test.csv")
X_test = test_data.drop(['mass'], axis=1)
Y_test = test_data['mass']

def getCorrelatedVariables():
    train_correlation_matrix = train_data.corr(method='pearson', min_periods=1, numeric_only=False)
    upper_triangle = np.triu(train_correlation_matrix, k=1)
    high_corr_pairs = np.where(upper_triangle >= 0.9)

    row_names = train_correlation_matrix.index[high_corr_pairs[0]]
    col_names = train_correlation_matrix.columns[high_corr_pairs[1]]

    high_corr_df = pd.DataFrame({
        'Var1': row_names,
        'Var2': col_names,
        'Correlation': train_correlation_matrix.values[high_corr_pairs]
    })


    return high_corr_df

def findSingleVariable():
    variable_names = X_train.columns

    # Values at index 0 are for variable at index 0 in variable_names, Values at index 1 ...
    r_squared_values = np.array([])
    p_values = np.array([])

    for variable in variable_names:
        x = sm.add_constant(X_train[variable])
        model = sm.OLS(Y_train, x).fit()
        r_squared_values = np.append(r_squared_values, model.rsquared)
        p_values = np.append(p_values, model.pvalues[1])

    # Filter out variables for which the p-value is larger than 0.05 (Null-hypothesis can't be rejected for those variables)
    indices_to_delete = np.where(p_values > 0.05)
    p_values = np.delete(p_values, indices_to_delete)
    r_squared_values = np.delete(r_squared_values, indices_to_delete)
    variable_names = np.delete(variable_names, indices_to_delete)

    highest_r2_index = np.argmax(r_squared_values)

    return variable_names[highest_r2_index], r_squared_values[highest_r2_index], p_values[highest_r2_index]

def multipleLinearRegression():
    train_data_x = X_train.drop(['Inj1PosVolAct_Var'], axis=1)

    variable_names = train_data_x.columns

    x = sm.add_constant(train_data_x)
    model = sm.OLS(Y_train, x).fit()


    # Remove predictor with highest p value
    pvals = model.pvalues[1:]
    index_to_remove = np.argmax(pvals)
    highest_p_val = pvals[index_to_remove]
    corresponding_variable = variable_names[index_to_remove]

    # Repeat until no more p-values above 0.05 or no more predictors
    while highest_p_val > 0.05 and len(variable_names) > 0:
        train_data_x = train_data_x.drop([corresponding_variable], axis=1)
        variable_names = train_data_x.columns

        x = sm.add_constant(train_data_x)
        model = sm.OLS(Y_train, x).fit()
        pvals = model.pvalues[1:]
        index_to_remove = np.argmax(pvals)
        highest_p_val = pvals[index_to_remove]
        corresponding_variable = variable_names[index_to_remove]

    return model.pvalues, model.rsquared, model, variable_names


def predictDatasets(model, relevant_variables):
    X_train_modified = sm.add_constant(X_train[relevant_variables])
    y_train_pred = model.predict(X_train_modified)
    mse_train = sm.tools.eval_measures.mse(Y_train, y_train_pred, axis=0)

    X_test_modified = sm.add_constant(X_test[relevant_variables])
    y_test_pred = model.predict(X_test_modified)
    mse_test = sm.tools.eval_measures.mse(Y_test, y_test_pred, axis=0)

    return mse_train, mse_test


def plotResiduals(model, relevant_variables, test=True):
    if test:
        X_modified = sm.add_constant(X_test[relevant_variables])
        Y = Y_test
        title_string = 'Residuals as a function of test predictions'
    else:
        X_modified = sm.add_constant(X_train[relevant_variables])
        Y = Y_train
        title_string = 'Residuals as a function of train predictions'

    y_pred = model.predict(X_modified)
    residuals = Y - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.title(title_string)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.show()


def higherOrderTerms(relevant_variables):
    train_data_x = X_train[relevant_variables]
    test_data_x = X_test[relevant_variables]

    # Exhaustive search + backward elimination
    for variable_1 in relevant_variables:
        for variable_2 in relevant_variables:
            # Skip if already present (other way around)
            if f'{variable_2}-{variable_1}' in train_data_x.columns:
                continue

            train_data_x[f'{variable_1}-{variable_2}'] = train_data_x[f'{variable_1}'] * \
                                                                                 train_data_x[f'{variable_2}']
            test_data_x[f'{variable_1}-{variable_2}'] = test_data_x[f'{variable_1}'] * \
                                                         test_data_x[f'{variable_2}']

    x = sm.add_constant(train_data_x)
    model = sm.OLS(Y_train, x).fit()

    #pvals = model.pvalues[1:]
    #index_to_remove = np.argmax(pvals)
    #highest_p_val = pvals[index_to_remove]
    #corresponding_variable = train_data_x.columns[index_to_remove]
    #while highest_p_val > 0.05:
    #    # Keep linear variables (hierarchical principle)
    #    if "-" not in corresponding_variable:
    #        pvals[index_to_remove] = 0
    #        index_to_remove = np.argmax(pvals)
    #        highest_p_val = pvals[index_to_remove]
    #        corresponding_variable = train_data_x.columns[index_to_remove]
    #        continue

    #    train_data_x = train_data_x.drop([corresponding_variable], axis=1)
    #    test_data_x = test_data_x.drop([corresponding_variable], axis=1)

    #    x = sm.add_constant(train_data_x)
    #    model = sm.OLS(Y_train, x).fit()
    #    pvals = model.pvalues[1:]
    #    index_to_remove = np.argmax(pvals)
    #    highest_p_val = pvals[index_to_remove]
    #    corresponding_variable = train_data_x.columns[index_to_remove]

    #x = sm.add_constant(train_data_x)
    #model = sm.OLS(Y_train, x).fit()


    y_train_pred = model.predict(x)
    mse_train = sm.tools.eval_measures.mse(Y_train, y_train_pred, axis=0)

    y_test_pred = model.predict(sm.add_constant(test_data_x))
    mse_test = sm.tools.eval_measures.mse(Y_test, y_test_pred, axis=0)

    return model.pvalues, model.rsquared, mse_train, mse_test


def main():
    # Answer to question 1:
    print("Question 1: ")
    print(getCorrelatedVariables())
    print('\n')

    # Answer to question 2:
    print("Question 2: ")
    variable_name, r2, p = findSingleVariable()
    print(variable_name+",", "R-squared: "+str(r2)+",", "p-value: "+str(p))
    print('\n')

    # Answer to question 3:
    print("Question 3: ")
    pvalues, r2, model, variable_names = multipleLinearRegression()
    print(pvalues)
    print("\nR-squared: "+str(r2)+"\n")

    # Answer to question 4:
    print("Question 4: ")
    mse_train, mse_test = predictDatasets(model, variable_names)
    print("Train MSE: "+str(mse_train)+"\nTest MSE: "+str(mse_test)+"\n")

    # Answer to question 5:
    print("Question 5: ")
    plotResiduals(model, variable_names, test=False)
    pvalues_higher_order, r2_higher_order, mse_train, mse_test = higherOrderTerms(variable_names)
    print(pvalues_higher_order)
    print("\nR-squared: "+str(r2_higher_order)+"\n")
    print("Train MSE: "+str(mse_train)+"\nTest MSE: "+str(mse_test)+"\n")

if __name__ == "__main__":
    main()