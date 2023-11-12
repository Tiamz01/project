import os
import sys
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error as MAE, mean_squared_error as MSE

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            loaded_file =dill.load(file_obj)
        return loaded_file 
    except Exception as e:
        raise CustomException("couldn't load file")

def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}
        

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})  # Get the hyperparameter grid for the current model

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)  # Set the best hyperparameters for the model
            model.fit(X_train, y_train)  # Train the model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = train_model_score
            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)