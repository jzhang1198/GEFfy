import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, Bounds
from scipy.stats import linregress
from matplotlib import pyplot as plt
import seaborn as sns

# 2DO: make the code more flexible by enabling usage of custom models
# Can include some in the library for the GEF-specific case
# Add a feature for collapsing plots

class GefFitter:
    """
    Class for fitting initial rates to GEF-catalyzed nucleotide 
    exchange progress curves AND for fitting Michaelis-Menten 
    constants to initial rate data.

    Attributes
    ----------
    """

    def __init__(self, data: pd.DataFrame, data_index: pd.DataFrame) -> None:
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
        self.headers = data.columns[1:]
        self.time = data.to_numpy(dtype=np.float64).T[0]
        self.ydatas = data.to_numpy(dtype=np.float64).T[1:]
        self.data_index = data_index # dynamically updated with fit parameters

        # initialize empty attributes to be assigned during fitting
        self.Km = float
        self.kcat = float

    def _map_sample_id(self, sample_id: str):
        row = self.data_index[self.data_index['sample'] == sample_id]

        if len(row) > 1:
            print('WARNING: Duplicate sample IDs detected')
            print('Exited from initialization')
            return 
        
        conc, GEF_conc, model, perc_curve, date = row.iloc[0]['conc'], row.iloc[0]['GEF_conc'], row.iloc[0]['model'], row.iloc[0]['perc_curve'], row.iloc[0]['date']
        return conc, GEF_conc, model, perc_curve, date
    
    @staticmethod   
    def _linear_model(time: np.ndarray, slope: float, yint: float):
        return (-slope * time) + yint

    @staticmethod   
    def _exponential_model_with_background(time: np.ndarray, span_exchange: float, k_exchange: float, span_background: float, k_background: float, fluorescence_plateau: float):
        return (span_exchange * np.exp(time * -k_exchange)) + (span_background * np.exp(time * -k_background)) + fluorescence_plateau

    @staticmethod   
    def _exponential_model(time: np.ndarray, span_exchange: float, k_exchange: float, fluorescence_plateau: float):
        return (span_exchange * np.exp(time * -k_exchange)) + fluorescence_plateau

    def _plot_progress_curves_and_fits(self, axs, xlabel, ylabel, palette):

        if not palette:
            palette = list(sns.color_palette("BuPu_r", len(self.ydatas)))
            palette.reverse()

        # re-format the data for seaborn plotting
        label_list, time_list, progress_list = [], [], []
        for index, label in enumerate(self.data_index['conc'].values.tolist()):
            progress_curve = self.ydatas[index]
            label_list += [label] * len(progress_curve)
            time_list += list(self.time) 
            progress_list += list(progress_curve)
        df = pd.DataFrame({'[S] (µM)': label_list, ylabel: progress_list, xlabel: time_list})
        
        # plot all progress curves on the main plot
        main_ax = axs.flatten()[0]
        sns.lineplot(ax=main_ax, data=df, x=xlabel, y=ylabel, hue='[S] (µM)', palette=palette)
        main_ax.set_title('Progress Curves')

        # plot progress curves individually with their fits
        for index, ax in enumerate(axs.flatten()[1:]):
            fit_summary = self.data_index.iloc[index]

            ax.set_title('{}; [S]={}µM'.format(fit_summary['sample'], fit_summary['conc']))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.plot(self.time, self.ydatas[index], color=palette[index])

            if fit_summary['model'] == 'linear_model':
                y_pred = GefFitter._linear_model(self.time, fit_summary['slope'], fit_summary['yint'])

            elif fit_summary['model'] == 'exponential_model':
                y_pred = GefFitter._exponential_model(self.time, fit_summary['span_exchange'], fit_summary['k_exchange'], fit_summary['fluorescence_plateau'])

            elif fit_summary['model'] == 'exponential_model_with_background':
                y_pred = GefFitter._exponential_model(self.time, fit_summary['span_exchange'], fit_summary['k_exchange'], fit_summary['fluorescence_plateau'])

            ax.plot(self.time, y_pred, color='black')

    def fit_initial_rates(self, 
                          plot=False, 
                          xlabel="Time (s)",
                          ylabel="Trp Fluorescence (RFUs)",
                          height_per_plot=10,
                          width_per_plot=10,
                          palette:list=None,
                          image_path:str=None):
        
        # parse through input data to obtain substrate and enzyme concentrations
        slopes, fluorescence_plateaus, k_exchanges, k_backgrounds, span_exchanges, span_backgrounds, pconvs, yints = [], [], [], [], [], [], [], []
        for header, ydata in zip(self.headers, self.ydatas):
            _, _, fit_type, perc_curve, _ = self._map_sample_id(header)

            if fit_type == 'linear_model':

                # truncate x and y data
                linear_regime_end = round(len(self.time) * perc_curve)
                time_trunc = self.time[0:linear_regime_end] 
                ydata_trunc = ydata[0:linear_regime_end]  

                # fit linear model
                bounds = Bounds(lb=np.array([0, 0]), ub=np.array([np.inf, np.inf]))
                popt, pconv = curve_fit(GefFitter._linear_model, time_trunc, ydata_trunc, bounds=(bounds.lb, bounds.ub))
                
                # unpack parameters and organize
                slope, yint = popt
                slopes.append(slope)
                fluorescence_plateaus.append(ydata[-1]) # this parameter is not fit, so just grab the last index of the ydata array
                k_exchanges.append('NA')
                k_backgrounds.append('NA')
                span_exchanges.append('NA')
                span_backgrounds.append('NA')
                yints.append(yint)
                pconvs.append(pconv)

            elif fit_type == 'exponential_model':

                # fit exponential model
                bounds = Bounds(np.array([0, 0, 0]), np.array([np.inf, np.inf, np.inf]))
                popt, pconv = curve_fit(GefFitter._exponential_model, self.time, ydata, bounds=(bounds.lb, bounds.ub))

                # unpack parameters and organize
                span_exchange, k_exchange, fluorescence_plateau = popt
                slopes.append('NA')
                fluorescence_plateaus.append(fluorescence_plateau)
                k_exchanges.append(k_exchange)
                k_backgrounds.append('NA')
                span_exchanges.append(span_exchange)
                span_backgrounds.append('NA')
                yints.append('NA')
                pconvs.append(pconv)            

            elif fit_type == 'exponential_model_with_background':

                # fit exponential model with background
                # for some reason, the fitting algorithm has trouble finding reasonable parameters without a good starting guess

                span_exchange_est = ydata.max() - ydata.min()
                k_exchange_est = 7e-3
                span_background_est = 0.2 * span_exchange_est
                k_background_est = 1e-4 
                fluorescence_plateau_est = ydata.min()

                bounds = Bounds(
                    lb=np.array([span_exchange_est - 0.05 * span_exchange_est, 5e-4, 0, 1e-5, fluorescence_plateau_est - 0.2 * fluorescence_plateau_est]), 
                    ub=np.array([span_exchange_est + 0.05 * span_exchange_est, 0.1, ydata.max(), 3e-4, fluorescence_plateau_est + 0.2 * fluorescence_plateau_est])
                    )
                
                initial_guess = np.array([
                    span_exchange_est,
                    k_exchange_est,
                    span_background_est,
                    k_background_est,
                    fluorescence_plateau_est
                ])
                
                popt, pconv = curve_fit(GefFitter._exponential_model_with_background, self.time, ydata, bounds=(bounds.lb, bounds.ub), maxfev=2000, p0=initial_guess)

                # unpack parameters and organize
                span_exchange, k_exchange, span_background, k_background, fluorescence_plateau = popt
                slopes.append('NA')
                fluorescence_plateaus.append(fluorescence_plateau)
                k_exchanges.append(k_exchange)
                k_backgrounds.append(k_background)
                span_exchanges.append(span_exchange)
                span_backgrounds.append(span_background)
                yints.append('NA')
                pconvs.append(pconv)      

            else:
                print(f'Fit type "{fit_type}" not recognized.')
                return 
            
        # update data index with summary of the fits
        self.data_index['slope'] = slopes
        self.data_index['yint'] = yints
        self.data_index['fluorescence_plateau'] = fluorescence_plateaus
        self.data_index['k_exchange'] = k_exchanges
        self.data_index['k_background'] = k_backgrounds
        self.data_index['span_exchange'] = span_exchanges
        self.data_index['span_background'] = span_backgrounds
        self.data_index['pconv'] = pconvs

        # 2DO: add some plotting functionality and attributes for storing fit statistics
        if plot:
            fig, axs = plt.subplots(nrows=len(self.ydatas) + 1, ncols=1, figsize=(width_per_plot, height_per_plot * len(self.ydatas) + 1))
            sns.set_style('ticks')
            self._plot_progress_curves_and_fits(axs, xlabel, ylabel, palette)

            if image_path:
                for index, ax in enumerate(axs.flatten()):
                    name = 'progress_curves' if index == 0 else ax.get_title().split(';')[0]
                    ax.figure.savefig(os.path.join(image_path, name + '.png'))

    def fit_conversion_factor(self, plot=False):
        """
        Fits a conversion factor to observed data. This will be used to 
        transform the initial rates from RFU/s to µM/s.

        This function is tailored specifically for fitting conversion factors
        for GEF-catalyzed exchange experiments. You may need to edit this function
        for your specific system. Alternatively, you could obtain a conversion 
        factor through construction of a standard curve.
        """

        concentration = self.data_index['conc'].to_numpy()
        fluorescence_plateau = self.data_index['fluorescence_plateau'].to_numpy()
        self.conversion_factor_fit = linregress(concentration, fluorescence_plateau)

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(concentration, fluorescence_plateau, color='black')
            x = np.linspace(0, concentration[-1], 1000)
            ax.plot(x, (self.conversion_factor_fit.slope * x) + self.conversion_factor_fit.intercept, '-')      
            ax.set_xlabel('Concentration (µM)')
            ax.set_ylabel('Trp Fluorescence (RFUs)')  
            print(self.conversion_factor_fit.rvalue)