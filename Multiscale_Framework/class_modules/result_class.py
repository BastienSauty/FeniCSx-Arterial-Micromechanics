import matplotlib.pyplot as plt
import numpy as np
import pickle

class Results:
    """
    This class handles the systematic storage and retrieval of simulation results, where the simulation is divided into 'nmax' equal steps.

    Parameters:
    -----------
    name : str
        The name identifier for the results, also used to generate the output filename.
    folder_name : str
        The name identifier for the results, also used to generate the output filename.
    dict_outputs : 
        dict containing for each key of output the points where these are computed. If 'points' is None then this is a global quantity    
    nmax : int
        The maximum number of iterations in the simulation.
    points : npoint*3 numpy array
        Optional : Contains coordinates of the points of interest. If not provided, stored quantities are just one scalar per output
        Otherwise, each output contains the value for each points at each step
    
    Attributes:
    -----------
    name : str
        The name identifier for the result set.
    param : dict or object
        Parameters used in the simulation.
    output_file : str
        The file path for saving the output data in a pickle file.
    outputs : dict
        A dictionary storing arrays of output values for each named output, initialized to zeros, with size `nmax+1`.
    
    Methods:
    --------
    export():
        Saves the current object instance to a pickle file.
    """

    def __init__(self, name, folder_name, dict_outputs, nmax, points=None):
        

        # Define the folder and filename for output data
        output_folder = "outputs"
        self.output_file = f"./{output_folder}/{folder_name}/{name}_scal.pkl" # output_folder + folder_name + '/'+ name + '_scal.pkl'

        # Initialize a dictionary to store output arrays, one for each named output
        self.runtime = 0
        self.outputs = {} # contain arrays of results
        self.dict_outputs = dict_outputs # contain the points where quantities are computed
        for key in dict_outputs:
            if dict_outputs[key]['points'] is None:                
                self.outputs[key] = np.zeros((nmax+1,))
            else:
                points = dict_outputs[key]['points']
                self.outputs[key] = np.zeros((nmax+1,len(points)))
            
    def export(self):
        """
        Save the current state of the object, including the simulation results and parameters, to a file in pickle format.
        This will overwrite any existing file with the same name.
        """
        with open(self.output_file, 'wb') as outp:
            pickle.dump(self, outp, -1)


def plot_results(results_list, keys_to_abscissa, keys_to_ordinate):
    """
    Plots the results from a list of `Results` objects, comparing specified outputs.

    Parameters:
    -----------
    results_list : list of Results objects
        A list containing instances of the `Results` class. Each object should have output data stored in the `outputs` attribute.
    keys_to_abscissa : list of str
        A list of keys corresponding to the outputs that will be used for the x-axis (abscissa) in the plot.
    keys_to_ordinate : list of str
        A list of keys corresponding to the outputs that will be used for the y-axis (ordinate) in the plot.

    Notes:
    ------
    - The lengths of `keys_to_abscissa` and `keys_to_ordinate` must be the same.
    - Each key in `keys_to_abscissa` and `keys_to_ordinate` must correspond to an output stored in the `outputs` dictionary of each `Results` object.
    - The function generates a plot with multiple curves, one for each result and ordinate/abscissa pair.
    """
    
    # Create a new figure and axis for plotting
    fig, ax = plt.subplots()

    # Loop through each result in the provided list of Results objects
    for result in results_list:
        if result:  # Ensure the result object is valid (not False or None)
            # Iterate over the list of ordinate keys and plot corresponding results
            for i in range(len(keys_to_ordinate)):
                # Plot the ordinate (y-axis) against the abscissa (x-axis) for the current result
                ax.plot(result.outputs[keys_to_abscissa[i]], result.outputs[keys_to_ordinate[i]],
                        label=f'{keys_to_ordinate[i]} - {result.name}')

    # Set x-axis label to the first key in `keys_to_abscissa`
    ax.set_xlabel(keys_to_abscissa[0])

    # Set y-axis label to the first key in `keys_to_ordinate`
    ax.set_ylabel(keys_to_ordinate[0])
    ax.legend()
    plt.grid()
    plt.show()
