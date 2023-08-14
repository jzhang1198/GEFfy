import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, Bounds
from scipy.stats import linregress
from matplotlib import pyplot as plt
import seaborn as sns
import os
import warnings

# Mute the SettingWithCopyWarning
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

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

    def __init__(self, data: str, data_index: str) -> None:
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
        self.headers = data.columns[1:].to_list()
        self.time = data.to_numpy(dtype=np.float64).T[0]
        self.ydatas = data.to_numpy(dtype=np.float64).T[1:]
        self.data_index = data_index # dynamically updated with fit parameters

        # initialize empty attributes to be assigned during fitting
        self.conversion_factor_fit = None
        self.Km = float
        self.kcat = float

    def _get_fit_inputs(self, sample_id: str):
        row = self.data_index[self.data_index['sample'] == sample_id]

        if len(row) > 1:
            print('WARNING: Duplicate sample IDs detected')
            print('Exited from initialization')
            return 
        
        conc, GEF_conc, model, perc_curve, date = row.iloc[0]['conc'], row.iloc[0]['GEF_conc'], row.iloc[0]['model'], row.iloc[0]['perc_curve'], row.iloc[0]['date']
        return conc, GEF_conc, model, perc_curve, date
    
    def _get_fit_outputs(self, sample_id: str):
        row = self.data_index[self.data_index['sample'] == sample_id]
        slope, F0, fluorescence_plateau, k_exchange, k_background, span_exchange, span_background, pconv, vF0 = row.iloc[0]['slope'], row.iloc[0]['F0'], row.iloc[0]['fluorescence_plateau'], row.iloc[0]['k_exchange'], row.iloc[0]['k_background'], row.iloc[0]['span_exchange'], row.iloc[0]['span_background'], row.iloc[0]['pconv'], row.iloc[0]['vF0']
        return slope, F0, fluorescence_plateau, k_exchange, k_background, span_exchange, span_background, pconv, vF0

    @staticmethod   
    def _linear_model(time: np.ndarray, slope: float, yint: float):
        return (-slope * time) + yint

    @staticmethod   
    def _exponential_model_with_background(time: np.ndarray, span_exchange: float, k_exchange: float, span_background: float, k_background: float, fluorescence_plateau: float):
        return (span_exchange * np.exp(time * -k_exchange)) + (span_background * np.exp(time * -k_background)) + fluorescence_plateau

    @staticmethod   
    def _exponential_model(time: np.ndarray, span_exchange: float, k_exchange: float, fluorescence_plateau: float):
        return (span_exchange * np.exp(time * -k_exchange)) + fluorescence_plateau

    @staticmethod
    def _michaelis_menten_model(substrate_concens: np.ndarray, Vmax: float, Km: float):
        return np.divide(Vmax * substrate_concens, Km + substrate_concens)

    def _plot_progress_curves_and_fits(self, axs, xlabel, ylabel, palette):

        if not palette:
            palette = list(sns.color_palette("BuPu_r", len(self.ydatas)))
            palette.reverse()

        # re-format the data for seaborn plotting
        label_list, time_list, progress_list = [], [], []
        for index, ax in enumerate(axs.flatten()[1:]):
            header = self.headers[index]
            conc, _, model, _, _ = self._get_fit_inputs(header)
            slope, F0, fluorescence_plateau, k_exchange, k_background, span_exchange, span_background, pconv, vF0 = self._get_fit_outputs(header)
            progress_curve = self.ydatas[self.headers.index(header)]

            # plot individual progress curve
            ax.set_title('{}; [S]={}µM'.format(header, conc))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.plot(self.time, progress_curve, color=palette[index])

            # plot fit to progress curve
            if model == 'linear_model':
                y_pred = GefFitter._linear_model(self.time, slope, F0)

            elif model == 'exponential_model':
                y_pred = GefFitter._exponential_model(self.time,  span_exchange, k_exchange, fluorescence_plateau)

            elif model == 'exponential_model_with_background':
                y_pred = GefFitter._exponential_model_with_background(self.time, span_exchange, k_exchange, span_background, k_background, fluorescence_plateau)

            ax.plot(self.time, y_pred, color='black')

            label_list += [conc] * len(progress_curve)
            time_list += list(self.time) 
            progress_list += list(progress_curve)

        df = pd.DataFrame({'[S] (µM)': label_list, ylabel: progress_list, xlabel: time_list})
        
        # plot all progress curves on the main plot
        main_ax = axs.flatten()[0]
        sns.lineplot(ax=main_ax, data=df, x=xlabel, y=ylabel, hue='[S] (µM)', palette=palette)
        main_ax.set_title('Progress Curves')

    def fit_initial_rates(self, 
                          initial_guess_and_constraints=False,
                          plot=False, 
                          xlabel="Time (s)",
                          ylabel="Trp Fluorescence (RFUs)",
                          height_per_plot=10,
                          width_per_plot=10,
                          palette:list=None,
                          image_path:str=None,
                          layout:str="portrait"):
        
        # parse through input data to obtain substrate and enzyme concentrations
        slopes, fluorescence_plateaus, k_exchanges, k_backgrounds, span_exchanges, span_backgrounds, pconvs, vF0s, F0s = [], [], [], [], [], [], [], [], []
        for header, ydata in zip(self.headers, self.ydatas):
            conc, _, fit_type, perc_curve, _ = self._get_fit_inputs(header)

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
                pconvs.append(pconv)
                vF0s.append(slope)
                F0s.append(yint)

            elif fit_type == 'exponential_model':

                # generate initial guess and constraints, if appropriate, and then fit
                if initial_guess_and_constraints:
                    span_exchange_est = ydata.max() - ydata.min()
                    k_exchange_est = 7e-3
                    fluorescence_plateau_est = ydata.min()
                    initial_guess = np.array([span_exchange_est, k_exchange_est, fluorescence_plateau_est])

                    bounds = Bounds(
                        lb=np.array([span_exchange_est - 0.5 * span_exchange_est, 5e-4, fluorescence_plateau_est - 0.2 * fluorescence_plateau_est]),
                        ub=np.array([span_exchange_est + 0.5 * span_exchange_est, 0.1, fluorescence_plateau_est + 0.2 * fluorescence_plateau_est])
                    )

                    popt, pconv = curve_fit(GefFitter._exponential_model, self.time, ydata, bounds=(bounds.lb, bounds.ub), p0=initial_guess)
                
                else:   
                    popt, pconv = curve_fit(GefFitter._exponential_model, self.time, ydata)

                # unpack parameters and organize
                span_exchange, k_exchange, fluorescence_plateau = popt
                slopes.append('NA')
                fluorescence_plateaus.append(fluorescence_plateau)
                k_exchanges.append(k_exchange)
                k_backgrounds.append('NA')
                span_exchanges.append(span_exchange)
                span_backgrounds.append('NA')
                pconvs.append(pconv)
                vF0s.append(span_exchange * k_exchange * np.exp(k_exchange * 0))     
                F0s.append(span_exchange + fluorescence_plateau)       

            elif fit_type == 'exponential_model_with_background':

                # fit exponential model with background
                # for some reason, the fitting algorithm has trouble finding reasonable parameters without a good starting guess
                # idea for automating initial guess estimation: compute derivative to identify inflection point of the function

                if initial_guess_and_constraints:
                    span_exchange_est = ydata.max() - ydata.min()
                    k_exchange_est = 7e-3
                    span_background_est = 0.2 * span_exchange_est
                    k_background_est = 1e-4 
                    fluorescence_plateau_est = ydata.min()
                    initial_guess = np.array([span_exchange_est, k_exchange_est, span_background_est, k_background_est, fluorescence_plateau_est])

                    bounds = Bounds(
                        lb=np.array([span_exchange_est - 0.05 * span_exchange_est, 5e-4, 0, 1e-5, fluorescence_plateau_est - 0.2 * fluorescence_plateau_est]), 
                        ub=np.array([span_exchange_est + 0.05 * span_exchange_est, 0.1, ydata.max(), 3e-4, fluorescence_plateau_est + 0.2 * fluorescence_plateau_est])
                    )

                    popt, pconv = curve_fit(GefFitter._exponential_model_with_background, self.time, ydata, bounds=(bounds.lb, bounds.ub), p0=initial_guess)

                else:
                    popt, pconv = curve_fit(GefFitter._exponential_model_with_background, self.time, ydata, maxfev=2000)

                # unpack parameters and organize
                span_exchange, k_exchange, span_background, k_background, fluorescence_plateau = popt
                slopes.append('NA')
                fluorescence_plateaus.append(fluorescence_plateau)
                k_exchanges.append(k_exchange)
                k_backgrounds.append(k_background)
                span_exchanges.append(span_exchange)
                span_backgrounds.append(span_background)
                pconvs.append(pconv)
                vF0s.append(span_exchange * k_exchange * np.exp(k_exchange * 0))     
                F0s.append(span_exchange + span_background + fluorescence_plateau)      

            else:
                print(f'Fit type "{fit_type}" not recognized.')
                return 
            
        # update data index with summary of the fits
        self.data_index['slope'] = slopes
        self.data_index['fluorescence_plateau'] = fluorescence_plateaus
        self.data_index['k_exchange'] = k_exchanges
        self.data_index['k_background'] = k_backgrounds
        self.data_index['span_exchange'] = span_exchanges
        self.data_index['span_background'] = span_backgrounds
        self.data_index['pconv'] = pconvs
        self.data_index['vF0'] = vF0s
        self.data_index['F0'] = F0s


        # 2DO: add some plotting functionality and attributes for storing fit statistics
        if plot:
            if layout == "portrait":
                fig, axs = plt.subplots(nrows=len(self.ydatas) + 1, ncols=1, figsize=(width_per_plot, height_per_plot * len(self.ydatas) + 1))
            elif layout == "landscape":
                fig, axs = plt.subplots(nrows=1, ncols=len(self.ydatas) + 1, figsize=(width_per_plot * len(self.ydatas) + 1, height_per_plot))
            else:
                print(f'ERROR: {layout} not a recognized value for layout parameter. Accepted values include: "portrait", "landscape".')
                                        
            sns.set_style('ticks')
            self._plot_progress_curves_and_fits(axs, xlabel, ylabel, palette)

            if image_path:
                plt.savefig(image_path, dpi=300)

    def fit_initial_rates_manual_guess(self, 
                          initial_guesses: pd.DataFrame,
                          plot=False, 
                          xlabel="Time (s)",
                          ylabel="Trp Fluorescence (RFUs)",
                          height_per_plot=10,
                          width_per_plot=10,
                          palette:list=None,
                          image_path:str=None,
                          layout:str="portrait"):
        
        # parse through input data to obtain substrate and enzyme concentrations
        slopes, fluorescence_plateaus, k_exchanges, k_backgrounds, span_exchanges, span_backgrounds, pconvs, vF0s, F0s = [], [], [], [], [], [], [], [], []
        for header, ydata in zip(self.headers, self.ydatas):
            conc, _, fit_type, perc_curve, _ = self._get_fit_inputs(header)

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
                pconvs.append(pconv)
                vF0s.append(slope)
                F0s.append(yint)

            elif fit_type == 'exponential_model':
                
                span_exchange_est = ydata.max() - ydata.min()
                k_exchange_est = 7e-3
                fluorescence_plateau_est = ydata.min()
                initial_guess = np.array([span_exchange_est, k_exchange_est, fluorescence_plateau_est])

                bounds = Bounds(
                    lb=np.array([span_exchange_est - 0.5 * span_exchange_est, 0, fluorescence_plateau_est - 0.2 * fluorescence_plateau_est])
                )

                popt, pconv = curve_fit(GefFitter._exponential_model, self.time, ydata, bounds=(bounds.lb, bounds.ub), p0=initial_guess)

                # unpack parameters and organize
                span_exchange, k_exchange, fluorescence_plateau = popt
                slopes.append('NA')
                fluorescence_plateaus.append(fluorescence_plateau)
                k_exchanges.append(k_exchange)
                k_backgrounds.append('NA')
                span_exchanges.append(span_exchange)
                span_backgrounds.append('NA')
                pconvs.append(pconv)
                vF0s.append(span_exchange * k_exchange * np.exp(k_exchange * 0))            
                F0s.append(span_exchange + fluorescence_plateau)

            elif fit_type == 'exponential_model_with_background':

                row = initial_guesses[initial_guesses['sample'] == header]
                span_exchange_est, k_exchange_est, span_background_est, k_background_est, fluorescence_plateau_est = row.iloc[0]['span_exchange_est'], row.iloc[0]['k_exchange_est'], row.iloc[0]['span_background_est'], 1e-4, row.iloc[0]['fluorescence_plateau_est']
                initial_guess = np.array([span_exchange_est, k_exchange_est, span_background_est, k_background_est, fluorescence_plateau_est])

                bounds = Bounds(
                    lb=np.array([span_exchange_est - 0.05 * span_exchange_est, 5e-4, 0, 1e-5, fluorescence_plateau_est - 0.2 * fluorescence_plateau_est]), 
                    ub=np.array([span_exchange_est + 0.05 * span_exchange_est, 0.1, ydata.max(), 3e-4, fluorescence_plateau_est + 0.2 * fluorescence_plateau_est])
                )

                popt, pconv = curve_fit(GefFitter._exponential_model_with_background, self.time, ydata, bounds=(bounds.lb, bounds.ub), p0=initial_guess)


                # unpack parameters and organize
                span_exchange, k_exchange, span_background, k_background, fluorescence_plateau = popt
                slopes.append('NA')
                fluorescence_plateaus.append(fluorescence_plateau)
                k_exchanges.append(k_exchange)
                k_backgrounds.append(k_background)
                span_exchanges.append(span_exchange)
                span_backgrounds.append(span_background)
                pconvs.append(pconv)
                vF0s.append(span_exchange * k_exchange * np.exp(k_exchange * 0))   
                F0s.append(span_exchange + span_background + fluorescence_plateau)       

            else:
                print(f'Fit type "{fit_type}" not recognized.')
                return 
            
        # update data index with summary of the fits
        self.data_index['slope'] = slopes
        self.data_index['fluorescence_plateau'] = fluorescence_plateaus
        self.data_index['k_exchange'] = k_exchanges
        self.data_index['k_background'] = k_backgrounds
        self.data_index['span_exchange'] = span_exchanges
        self.data_index['span_background'] = span_backgrounds
        self.data_index['pconv'] = pconvs
        self.data_index['vF0'] = vF0s
        self.data_index['F0'] = F0s


        # 2DO: add some plotting functionality and attributes for storing fit statistics
        if plot:
            if layout == "portrait":
                fig, axs = plt.subplots(nrows=len(self.ydatas) + 1, ncols=1, figsize=(width_per_plot, height_per_plot * len(self.ydatas) + 1))
            elif layout == "landscape":
                fig, axs = plt.subplots(nrows=1, ncols=len(self.ydatas) + 1, figsize=(width_per_plot * len(self.ydatas) + 1, height_per_plot))
            else:
                print(f'ERROR: {layout} not a recognized value for layout parameter. Accepted values include: "portrait", "landscape".')
                                        
            sns.set_style('ticks')
            self._plot_progress_curves_and_fits(axs, xlabel, ylabel, palette)

            if image_path:
                plt.savefig(image_path, dpi=300)

    def fit_conversion_factor(self, plot:bool=False, image_path:str=None):
        """
        Fits a conversion factor to observed data. This will be used to 
        transform the initial rates from RFU/s to µM/s.

        This function is tailored specifically for fitting conversion factors
        for GEF-catalyzed exchange experiments. You may need to edit this function
        for your specific system. Alternatively, you could obtain a conversion 
        factor through construction of a standard curve.
        """

        concentrations = self.data_index['conc'].to_numpy()

        try: 
            F0s = self.data_index['F0'].to_numpy()
        except:
            print('No values for F0 fit. Please fit initial rates before fitting conversion factors.')
            return

        self.conversion_factor_fit = linregress(F0s, concentrations)

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(F0s, concentrations, color='black')
            x = np.linspace(0, max(F0s), 1000)
            ax.plot(x, (self.conversion_factor_fit.slope * x) + self.conversion_factor_fit.intercept, color='blue')      
            ax.set_ylabel('Concentration (µM)')
            ax.set_xlabel('Trp Fluorescence (RFUs)')  
            text = '\n'.join([
                'm = {:.2e}'.format(self.conversion_factor_fit.slope),
                '$y_{int}$ ' + '= {:.2f}'.format(self.conversion_factor_fit.intercept),
                '$R^2$ = {:.2f}'.format(self.conversion_factor_fit.rvalue)
                ])
            ax.text(0.05 * max(x), .8 * max(concentrations), text, fontsize=12)

            if image_path:
                plt.savefig(image_path, dpi=300)

    def fit_michaelis_menten(self, plot:bool=False, image_path:str=None, title:str='GEF-catalyzed Exchange Michaelis-Menten Curve'):

        if not self.conversion_factor_fit or 'vF0' not in self.data_index.columns:
            print('ERROR: Conversion factors or initial velocities have not yet been fit.')
            return
        
        initial_velocities = self.data_index['vF0'].to_numpy() * self.conversion_factor_fit.slope # convert to molar concentration
        substrate_concens = self.data_index['conc'].to_numpy() 
        gef_concs = list(set(self.data_index['GEF_conc']))
        gef_conc = gef_concs.pop()

        # prepend zeros to substrate and initial velocity arrays
        initial_velocities = np.concatenate([[0], initial_velocities])
        substrate_concens = np.concatenate([[0], substrate_concens])

        if len(gef_concs) > 1:
            print('ERROR: [GEF] varies between experimental groups. You will need to separate progress curves into groups of the same [GEF].')
            return 

        popt, pconv = curve_fit(GefFitter._michaelis_menten_model, substrate_concens, initial_velocities)
        Vmax, Km = popt 
        self.kcat = Vmax / (gef_conc / 1000)
        self.Km = Km

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(substrate_concens, initial_velocities / (gef_conc / 1000), color='black')
            x = np.linspace(0, substrate_concens[-1] * 2, 1000)
            ax.plot(x, GefFitter._michaelis_menten_model(x, Vmax, Km) / (gef_conc / 1000), '-')      
            ax.set_xlabel('[S] (µM)')
            ax.set_ylabel('Enzyme-Normalized V0 ($s^{-1}$)')  

            title = '\n'.join([
                title,
                '$k_{cat}$ = ' + '{:.2f}'.format(self.kcat) + ', $K_m$ = ' + '{:.2f}'.format(self.Km)
            ])
            ax.set_title(title)

            if image_path:
                plt.savefig(image_path, dpi=300)