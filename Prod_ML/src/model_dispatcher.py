#Imports
from sklearn import linear_model,tree,ensemble,neural_network

#List
models={
    # "string_name":model(),
    "linear_regression":linear_model.LinearRegression(),
    "decision_tree":tree.DecisionTreeRegressor(),
    "random_forest":ensemble.RandomForestClassifier(),
    "mlp":neural_network.MLPRegressor()
}