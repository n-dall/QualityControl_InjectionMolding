import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

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
    # Alternative Strategy: Pick the variable that has the highest correlation coefficient with 'mass'

    # Get all variable names and remove mass from array
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

    # Pick highest r2
    highest_r2_index = np.argmax(r_squared_values)

    return variable_names[highest_r2_index], r_squared_values[highest_r2_index], p_values[highest_r2_index]

def multipleLinearRegression(var_to_drop = 'Inj1PosVolAct_Var'):
    if var_to_drop == 'None':
        train_data_x = X_train
    elif var_to_drop == 'Inj1PrsAct_meanOfInjPhase':
        train_data_x = X_train.drop(['Inj1PrsAct_meanOfInjPhase'], axis=1)
    else:
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

def pairPlotRelevantVariables(relevant_variables):
    # For plotting all relevant variables against all others
    relevant_x_y = train_data.drop([variable for variable in train_data.columns if variable not in relevant_variables and variable != 'mass'], axis=1)
    sns.pairplot(relevant_x_y)
    matplotlib.pyplot.show()

def plotResiduals(model, relevant_variables, test=True):
    if test:
        X_test_modified = sm.add_constant(X_test[relevant_variables])
        y_test_pred = model.predict(X_test_modified)

        residuals = Y_test - y_test_pred

        matplotlib.pyplot.figure()
        matplotlib.pyplot.scatter(y_test_pred, residuals)
        matplotlib.pyplot.title('Residuals as a function of test predictions')
        matplotlib.pyplot.xlabel('Predictions')
        matplotlib.pyplot.ylabel('Residuals')
        matplotlib.pyplot.show()
    else:
        X_train_modified = sm.add_constant(X_train[relevant_variables])
        y_train_pred = model.predict(X_train_modified)

        residuals = Y_train - y_train_pred

        matplotlib.pyplot.figure()
        matplotlib.pyplot.scatter(y_train_pred, residuals)
        matplotlib.pyplot.title('Residuals as a function of train predictions')
        matplotlib.pyplot.xlabel('Predictions')
        matplotlib.pyplot.ylabel('Residuals')
        matplotlib.pyplot.show()


def higherOrderTerms(relevant_variables):
    # Relevant variables: Inj1PrsAct_meanOfInjPhase, Inj1HtgEd3Act_1stPCscore, ClpFceAct_1stPCscore, OilTmp1Act_1stPCscore

    train_data_x = X_train[relevant_variables]
    test_data_x = X_test[relevant_variables]

    # These together bump R2 from 0.9818914389632266 to 0.9825576670684059 (second one not really relevant)
    # But they increase test MSE
    # train_data_x['Inj1PrsAct_meanOfInjPhase_SQUARED'] = train_data_x['Inj1PrsAct_meanOfInjPhase'] ** 2
    # test_data_x['Inj1PrsAct_meanOfInjPhase_SQUARED'] = test_data_x['Inj1PrsAct_meanOfInjPhase'] ** 2
    # train_data_x['Inj1PrsAct_meanOfInjPhase-Inj1HtgEd3Act_1stPCscore'] = train_data_x['Inj1PrsAct_meanOfInjPhase'] * train_data_x['Inj1HtgEd3Act_1stPCscore']

    # Exhaustive search for quadratic/interaction terms
    highest_r2 = 0
    var_1 = ''
    var_2 = ''
    for variable_1 in relevant_variables:
        for variable_2 in relevant_variables:
            train_data_x = X_train[relevant_variables]
            train_data_x[f'{variable_1}-{variable_2}'] = train_data_x[f'{variable_1}'] * \
                                                                                 train_data_x[f'{variable_2}']
            x = sm.add_constant(train_data_x)
            model = sm.OLS(Y_train, x).fit()

            if model.rsquared > highest_r2:
                highest_r2 = model.rsquared
                var_1 = variable_1
                var_2 = variable_2

    train_data_x = X_train[relevant_variables]
    train_data_x[f'{var_1}-{var_2}'] = train_data_x[f'{var_1}'] * train_data_x[f'{var_2}']
    test_data_x[f'{var_1}-{var_2}'] = test_data_x[f'{var_1}'] * test_data_x[f'{var_2}']

    x = sm.add_constant(train_data_x)
    model = sm.OLS(Y_train, x).fit()


    y_train_pred = model.predict(x)
    mse_train = sm.tools.eval_measures.mse(Y_train, y_train_pred, axis=0)

    y_test_pred = model.predict(sm.add_constant(test_data_x))
    mse_test = sm.tools.eval_measures.mse(Y_test, y_test_pred, axis=0)

    return model.pvalues, model.rsquared, mse_train, mse_test


def main():
    # Answer to question 1:
    #print("Question 1: ")
    #print(getCorrelatedVariables())
    #print('\n')

    # Answer to question 2:
    #print("Question 2: ")
    #variable_name, r2, p = findSingleVariable()
    #print(variable_name+",", "R-squared: "+str(r2)+",", "p-value: "+str(p))
    #print('\n')

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
    # pairPlotRelevantVariables(variable_names)
    # plotResiduals(model, variable_names, test=False)
    pvalues_higher_order, r2_higher_order, mse_train, mse_test = higherOrderTerms(variable_names)
    print(pvalues_higher_order)
    print("\nR-squared: "+str(r2_higher_order)+"\n")
    print("Train MSE: "+str(mse_train)+"\nTest MSE: "+str(mse_test)+"\n")

if __name__ == "__main__":
    main()