"""
Created by Andy S Maxwell 04/03/2024
Helper functions for plotting related to housepricer
"""

import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


#Helper function for making plots of true vs predicted values
def plot_cross_validated_pred(y:list, y_pred:list, filename : str = None) -> None:
    """
    Function to use matlib plot to show predicted vs true y and the residuals
    """
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    if filename == None:
        plt.show()
    else:
        print(filename)
        plt.savefig(filename, format='png')