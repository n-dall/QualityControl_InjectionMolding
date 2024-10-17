# Description
Predicts mass of ice scrapers created via injection molding based on various variables that are measured during production. (Mass is the most important measure to predict the quality of an ice scraper)

# Answered questions
1. Analyze the training data. Is there any variable which is highly correlated to another variable? List all
variables with correlation coefficients ≥ 0.9.
2. Assume you can only choose one feature to predict the mass as well as possible. Which variable do
you select? Explain why you select this variable and show the relevant numbers (p-value and R2-value).
3. Build a linear regression model which uses as many input variables as required. Keep in mind that
each sensor costs money, so remove variables which are not needed from the model. List the
selected variables and the relevant numbers for selecting them (p-value and R2-value).
4. Use the selected model to predict the mass on the test data. Compare the training MSE to the test
MSE. How does your model perform on the test data?
5. Add higher-order terms, such as quadratic terms or interaction terms to improve the model. Judge
the model quality (R2 values). Again, compare the training MSE to the test MSE.
