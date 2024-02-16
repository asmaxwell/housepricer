"""
Created by Andy S Maxwell 14/02/2024
Provide app interface for housepricer
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from housepricer import modeler



def full_run_evolution(data_directory: str
                       , model_directory : str
                       , postcode_directory :str 
                       , population_size : int
                       ,  generations : int) -> None:
    """Full set up, hyperparameter search and training of model, which is then saved to file
        Uses evolution algorithm
        To be run from command line via __main__.py
    """
       
    print("loading data and setting up model")
    model = modeler.random_forest(data_directory, model_directory, None, postcode_directory)

    print("doing evolution hyperparameter search")
    model.evolve_hyperparameter_search(population_size, generations)

    print(model.model)
    model.save_model("EvolveModel.save")

    print("saving True vs Pred")
    model.save_true_vs_predicted("True_Pred.png")

    return

def full_run_random(data_directory: str
                       , model_directory : str
                       , postcode_directory :str 
                       , iterations : int) -> None:
    """Full set up, hyperparameter search and training of model, which is then saved to file
        Uses random search
        To be run from command line via __main__.py
    """
    # initiate model
       
    print("loading data and setting up model")
    model = modeler.random_forest(data_directory, model_directory, None, postcode_directory)

    print("doing evolution hyperparameter search")
    model.random_hyperparameter_search(iterations)

    print(model.model)
    model.save_model("RandomModel.save")

    print("saving True vs Pred")
    model.save_true_vs_predicted("True_Pred.png")

    return 

def cal_run_evolution(model_directory, population_size : int
                       ,  generations : int) -> None:
    """Using californian dataset, hyperparameter search and training of model, which is then saved to file
        Uses evolution algorithm
        To be run from command line via __main__.py
    """
       
    print("loading data and setting up model")
    model = modeler.random_forest(None, model_directory, None, None, True)

    print("doing evolution hyperparameter search")
    model.evolve_hyperparameter_search(population_size, generations)

    print(model.model)
    model.save_model("EvolveModel.save")

    print("saving True vs Pred")
    model.save_true_vs_predicted("True_Pred.png")

    return

def cal_run_random(model_directory, iterations : int) -> None:
    """Using californian dataset, hyperparameter search and training of model, which is then saved to file
        Uses random search
        To be run from command line via __main__.py
    """
    # initiate model
       
    print("loading data and setting up model")
    model = modeler.random_forest(None, model_directory, None, None, True)

    print("doing evolution hyperparameter search")
    model.random_hyperparameter_search(iterations)

    print(model.model)
    model.save_model("RandomModel.save")

    print("saving True vs Pred")
    model.save_true_vs_predicted("True_Pred.png")

    return 

    