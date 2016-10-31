 def validation(model, X_train, y_train):
        
    #Set the parameters by cross-validation
    param_grid = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,), (60,)],
    'alpha': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]}

    validated = GridSearchCV(model, param_grid, CV = 5)
    validated.fit(X_train, y_train)

    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    

    return validated


