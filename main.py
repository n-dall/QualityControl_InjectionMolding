import pandas as pd
import numpy as np
import statsmodels.api as sm

train_data = pd.read_csv("InjectionMolding_Train.csv")
test_data = pd.read_csv("InjectionMolding_Test.csv")

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
    variable_names = train_data.columns[:8]

    # Values at index 0 are for variable at index 0 in variable_names, Values at index 1 ...
    r_squared_values = np.array([])
    p_values = np.array([])

    for variable in variable_names:
        x = sm.add_constant(train_data[variable])
        model = sm.OLS(train_data['mass'], x).fit()
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

if __name__ == "__main__":
    main()