from sklearn.model_selection import GridSearchCV

def validation(model, X_train, y_train, param_grid):

    validated = GridSearchCV(model, param_grid, cv = 5)
    validated.fit(X_train, y_train)

    print("Best parameters set found on training set:")
    print()
    print(validated.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = validated.cv_results_['mean_test_score']
    stds = validated.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, validated.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))


    return validated
