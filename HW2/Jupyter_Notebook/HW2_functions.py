def add_dummy_variable(df, var, dummy_var, lambda_equation):
    df[dummy_var] = df[var].apply(lambda_equation)


def data_split(df, var, test_size):
    X = df
    Y = df[var]
    test_size = 0.3
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test
