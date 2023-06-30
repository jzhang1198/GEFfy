import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import seaborn as sns

class GefFitter:
    """
    Class for fitting initial rates to GEF-catalyzed nucleotide 
    exchange progress curves AND for fitting Michaelis-Menten 
    constants to initial rate data.

    Attributes
    ----------

    """

    def __init__(self, data_file: str, data_index_file: str) -> None:
        """
        data_file should be a path to a .csv file. The code 
        assumes that the first column within this file is time.
        All subsequent columns are assumed to be fluorescence
        intensity measurements for different samples. Samples 
        should be named with the following convention:

        <GTPase-variant-name>_<index>

        data_index_file also should be a path to a .csv file. This 
        file contains relevant experimental information associated 
        with each sample. Rows are indexed by sample. This file should 
        contain five columns:

        1) "sample": sample names. These should map to columns in data_file.
        2) "conc": the concentration of GTPase in the sample in units of µM.
        3) "GEF_conc": the concentration of GEF in the sample in units of nM.
        4) "fit_type": determines the type of fit. Either "linear" or "exp".
        5) "perc_curve": defines the percentage of the progress curve to fit
           a line to. Ignored if "fit_type" is set to exp.
        6) "background_fit": determines whether a term accounting for
           background photobleaching. 1 for inclusion, 0 for exclusion.
        7) "date": contains dates in YYYYMMDD format.
        """

        # get data
        _data = pd.read_csv(data_file)
        self.headers = _data.columns
        self.time = _data.to_numpy()[0]
        self.ydatas = _data.to_numpy()[1:]
        self._data_index = pd.read_csv(data_index_file)

        # initialize empty attributes to be assigned during fitting
        self.substrate_concentrations = np.ndarray
        self.initial_rates = np.ndarray
        self.Km = float
        self.kcat = float

    def _map_sample_id(self, sample_id: str):
        row = self._data_index[self._data_index['sample'] == sample_id]

        if len(row) > 1:
            print('WARNING: Duplicate sample IDs detected')
            print('Exited from initialization')
            return 
        
        return row['conc'][0], row['GEF_conc'][0], row['fit_type'][0], row['perc_curve'][0],row['background_fit'][0], row['date'][0]
    
    @staticmethod
    def _fit_linear_with_background(time: np.ndarray, ydata: np.ndarray):

        # define the model to fit the data to and the objective function for NLS
        def model(time, slope, span_background, k_background, fluorescence_plateau):
            return (slope * time) + (span_background * np.exp(time * k_background)) + fluorescence_plateau
            
        # run NLS with scipy.optimize.curve_fit()
        popt, pconv = curve_fit(model, time, ydata)
        slope, span_background, k_background, fluorescence_plateau = tuple(popt)
        return slope, span_background, k_background, fluorescence_plateau

    @staticmethod
    def _fit_linear(time: np.ndarray, ydata: np.ndarray):
        # define the model to fit the data to and the objective function for NLS
        def model(time, slope, fluorescence_plateau):
            return (slope * time) + fluorescence_plateau
            
        # run NLS with scipy.optimize.curve_fit()
        popt, pconv = curve_fit(model, time, ydata)
        slope, fluorescence_plateau = tuple(popt)
        return slope, fluorescence_plateau            

    @staticmethod
    def _fit_exponential_with_background(time: np.ndarray, ydata: np.ndarray):

        # define the model to fit the data to and the objective function for NLS
        def model(time, span_exchange, k_exchange, span_background, k_background, fluorescence_plateau):
            return (span_exchange * np.exp(time * k_exchange)) + (span_background * np.exp(time * k_background)) + fluorescence_plateau
            
        # run NLS with scipy.optimize.curve_fit()
        popt, pconv = curve_fit(model, time, ydata)
        span_exchange, k_exchange, span_background, k_background, fluorescence_plateau = tuple(popt)
        return span_exchange, k_exchange, span_background, k_background, fluorescence_plateau

    @staticmethod
    def _fit_exponential(time: np.ndarray, ydata: np.ndarray):

        # define the model to fit the data to and the objective function for NLS
        def model(time, span_exchange, k_exchange, fluorescence_plateau):
            return (span_exchange * np.exp(time * k_exchange)) + fluorescence_plateau
            
        # run NLS with scipy.optimize.curve_fit()
        popt, pconv = curve_fit(model, time, ydata)
        span_exchange, k_exchange, fluorescence_plateau = tuple(popt)
        return span_exchange, k_exchange, fluorescence_plateau

    def fit_initial_rates(self, 
                          plot_progress_curves=False, 
                          plot_fits=False, 
                          title="Progress Curves",
                          hue=None,
                          xlabel="Time (s)",
                          ylabel="Trp Fluorescence (RFUs)"):
        
        # parse through input data to obtain substrate and enzyme concentrations
        substrate_concentrations = []
        initial_rates = []
        for header, ydata in zip(self.headers, self.ydatas):
            gtpase_conc, gef_conc, fit_type, perc_curve, background_fit, date = self._map_sample_id(header)

            if fit_type == 'linear':
                if background_fit == 1:
                    slope, span_background, k_background, fluorescence_plateau = GefFitter._fit_linear_with_background(self.time, ydata)
                    initial_rate = slope
                elif background_fit == 0:
                    slope, fluorescence_plateau = GefFitter._fit_linear(self.time, ydata)
                    initial_rate = slope
                else:
                    print(f'Background fit "{background_fit}" not recognized')
                    return

            elif fit_type == 'exp':
                if background_fit == 1:
                    span_exchange, k_exchange, span_background, k_background, fluorescence_plateau = GefFitter._fit_exponential_with_background(self.time, ydata)
                    initial_rate = k_exchange
                elif background_fit == 0:
                    span_exchange, k_exchange, fluorescence_plateau = GefFitter._fit_exponential(self.time, ydata)
                else:
                    print(f'Background fit "{background_fit}" not recognized')
                    return

            else:
                print(f'Fit type "{fit_type}" not recognized.')
                return 
            
            substrate_concentrations.append(gtpase_conc)
            initial_rates.append(initial_rate / (gef_conc / 1000)) # normalize initial rate by [E]

        self.substrate_concentrations = np.array(substrate_concentrations)
        self.initial_rates = np.array(initial_rates)

        # 2DO: add some plotting functionality and attributes for storing fit statistics

        return

    def fit_michaelis_menten(self):

        if not self.initial_rates:
            print("No initial rate data has been fit!")
            print("Obtain initial rates by running the fit_initial_rates method.")
            return
        
        def michaelis_menten_model(substrate_concentrations: np.ndarray, Km: float, Vmax: float):
            return np.divide(Vmax * substrate_concentrations, substrate_concentrations + Km)

        popt, pconv = curve_fit(michaelis_menten_model)
        
        return