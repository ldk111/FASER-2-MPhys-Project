from scipy.optimize import minimize
import pandas as pd
import numpy as np

def Objective_Function(parameters, untransformed_data, transformed_data):
    """
    Objective function to minimize. Calculates the residuals between
    the actual transformed data and the predicted transformed data.

    Parameters:
    - parameters: Transformation parameters to be optimized.
    - untransformed_data: The original untransformed data.
    - transformed_data: The actual transformed data.

    Returns:
    - residuals: The sum of squared residuals.
    """
    # Apply the transformation to untransformed data using the parameters
    predicted_transformed_data = Transformation_Function(parameters, untransformed_data)

    # Calculate the residuals (difference between actual and predicted transformed data)
    residuals = transformed_data[0] - predicted_transformed_data[0]
    residuals = np.append(residuals, transformed_data[1] - predicted_transformed_data[1])

    # Return the sum of squared residuals
    return np.sum(residuals**2)

def Transformation_Function(parameters, untransformed_data):
    """
    Define your transformation function here using the given parameters.
    This function should perform the transformation on the untransformed data.

    Parameters:
    - parameters: Transformation parameters [tx, ty, tz, dx, dy, dz].
    - untransformed_data: The original untransformed data.

    Returns:
    - transformed_data: The data after applying the transformation.
    """

    cx = np.cos(parameters[0])
    sx = np.sin(parameters[0])

    cy = np.cos(parameters[1])
    sy = np.sin(parameters[1])

    cz = np.cos(parameters[2])
    sz = np.sin(parameters[2])

    dx = parameters[3]
    dy = parameters[4]
    dz = parameters[5]

    #Apply transformation parameterised by the 3 rotation angles and 3 offsets
    transformed_data_y = (cz*cx - sz*sy*sx)*(untransformed_data[0] + dy) - sx*cy*(untransformed_data[1] + dz) + (sz*cx + cz*sx*sy) * dx
    transformed_data_z = (cz*sx + cx*sz*sy)*(untransformed_data[0] + dy) + cx*cy*(untransformed_data[1] + dz) + (sz*sx - cx*cz*sy) * dx

    transformed_data = np.array([transformed_data_y, transformed_data_z])

    return transformed_data

def Find_Best_Transform(untransformed_data, transformed_data, initial_guess, bounds):
    """
    Find the transformation parameters that minimize the residuals.

    Parameters:
    - untransformed_data: The original untransformed data.
    - transformed_data: The actual transformed data.
    - initial_guess: Initial guess for the transformation parameters.

    Returns:
    - result: The result object returned by scipy's minimize function.
    """

    result = minimize(Objective_Function, initial_guess, args=(untransformed_data, transformed_data), bounds=bounds)

    return result


def Fit_Offsets(df, plane, initial_guess = np.random.uniform(-0.1, 0.1, 6), bounds = [(-np.pi/8, np.pi/8),(-np.pi/8, np.pi/8),(-np.pi/8, np.pi/8),(-1, 1),(-1, 1),(-1, 1)]):
    """
    Find the transformation parameters that minimize the residuals for the given tracking plane and return result and goodness of fit.

    Parameters:
    - df: The offset dataframe produced from the simulation, containing the measured hit and predicted hit coordinates.
    - plane: Which tracking plane we are looking at (1-6).
    - initial_guess: Initial guess for the transformation parameters.
    - bounds: Lower and upper bounds for each of the parameters [tx, ty, tz, dx, dy, dz], can set to (0,0) if you want to eliminate a transformation.

    Returns:
    - result.x: The fitted parameters returned by Find_Best_Transform.
    - sum_of_residuals_squared: Sum of squared residuals between the transformed predicted hit location and measured hit location.
    """
    #This is the predicted location assuming no offsets
    untransformed_data = np.array([df["PRED_Y_" + str(plane)].values, df["PRED_Z_" + str(plane)].values])

    #This is the measured hit on the offset geometry
    transformed_data = np.array([df["HIT_Y_"+ str(plane)].values, df["HIT_Z_"+ str(plane)].values])

    result = Find_Best_Transform(untransformed_data, transformed_data, initial_guess, bounds)

    sum_of_residuals_squared = Objective_Function(result.x, untransformed_data, transformed_data)
    
    print("Optimal transformation parameters:", result.x)
    print("Sum of squares of residuals:", sum_of_residuals_squared)
    print("Sum of squares of residuals (no transform):", Objective_Function([0,0,0,0,0,0], untransformed_data, transformed_data))
    print("\n")

    return result.x, sum_of_residuals_squared

