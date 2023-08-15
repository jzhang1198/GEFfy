import os
import pandas as pd
import numpy as np
import seaborn as sns
import ipywidgets as widgets
from scipy.stats import linregress, chisquare
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, Bounds
from sklearn.linear_model import LinearRegression
from IPython.display import display, clear_output

# mute annoying pandas warnings
import warnings
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)


# 2DO: make the code more flexible by enabling usage of custom models
# Integrate manual initial guess as a kwarg instead of a separate function

def divide_chunks(l, n):
    """
    Taken from the kinetic_analysis library: 
    https://github.com/pinneylab/kinetic_analysis/tree/main

    Function to split a list 'l' into 'n' equal-sized chunks.
    The function yields each chunk as a separate list.
    It ensures that no chunk is larger than the original list.
    """

    chunk_size = len(l) // n
    remainder = len(l) % n
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size
        yield l[start:end]
        start = end

def compute_initial_reaction_slope(time_arr: np.array, 
                                   signal_arr: np.array,
                                  min_included_percent: int = 2): 
    """
    Taken from the kinetic_analysis library: 
    https://github.com/pinneylab/kinetic_analysis/tree/main

    Determines best linear fit to initial slope of data, by fitting regressions to 
    all percentiles of the data--greater than some defined minimum--anchored at the origin.
    
    Args:
        time_arr (np.array): array of assay read times
        signal_arr (np.array): array of kinetic signal readouts
        min_included_percent (int) = 5: minimum percent of data to be included 
    
    Returns:
        slope, intercept, score ((float, float, float)): fit parameters of best fit
    """

    # Need to triage further for different definitions of minimum
    MIN_INCLUDED_DATA_POINTS = 3
    
    perc_concs = list(divide_chunks(signal_arr, 100))
    perc_times = list(divide_chunks(time_arr, 100))
 
    scores = []
    slopes = [] 
    intercepts = []
    
    min_inclusion = max(len(perc_times) * min_included_percent // 100, MIN_INCLUDED_DATA_POINTS)
    
    #for i in range(len(perc_times), min_inclusion, -1):
    for i in range(min_inclusion, len(perc_times), 1):
        curr_times = np.concatenate(perc_times[:i])
        curr_concs = np.concatenate(perc_concs[:i])
        reg = LinearRegression().fit(np.array(curr_times).reshape(-1,1), curr_concs)
        curr_score = reg.score(np.array(curr_times).reshape(-1,1), curr_concs)
        scores.append(curr_score)
        slopes.append(reg.coef_)
        intercepts.append(reg.intercept_)
    
    if len(scores) == 0 and min_included_percent == 100:
        reg = LinearRegression().fit(np.array(perc_times).reshape(-1,1), perc_concs)
        curr_score = reg.score(np.array(perc_times).reshape(-1,1), perc_concs)
        return reg.coef_.item(), reg.intercept_.item(), curr_score

    # The fit with the highest R-squared value is selected as the best fit.
    max_r2_idx = np.argmax(scores)
    
    if scores[max_r2_idx] < 0.9:
        return np.array([np.nan]), np.nan, np.array([np.nan])
    
    return slopes[max_r2_idx], intercepts[max_r2_idx], scores[max_r2_idx]

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
        self.current_index= 0 # attribute for interactive plotting

        # initialize empty attributes to be assigned during fitting
        self.conversion_factor_fit = None
        self.Km = float
        self.kcat = float

        self.Km_sse_squared_array = np.ndarray
        self.Km_array = np.ndarray
        self.kcat_sse_squared_array = np.ndarray
        self.kcat_array = np.ndarray

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

    def _plot_progress_curves_and_fits(self, xlabel, ylabel, palette, width_per_plot, height_per_plot):
        
        plot_data = []
        if not palette:
            palette = list(sns.color_palette("BuPu_r", len(self.ydatas)))
            palette.reverse()

        colors, ydata, labels = [], [], []
        for index, header in enumerate(self.headers):
            conc, _, model, _, _ = self._get_fit_inputs(header)
            slope, F0, fluorescence_plateau, k_exchange, k_background, span_exchange, span_background, pconv, vF0 = self._get_fit_outputs(header)
            progress_curve = self.ydatas[self.headers.index(header)]

            title = '{}; [S]={:.2f}µM; {}'.format(header, conc, model) + '\n$vF_{0} = $' + '{:.2f}'.format(vF0)
            color = palette[index]
            label = '{:.2f} µM'.format(conc)

            # plot fit to progress curve
            if model == 'linear_model':
                y_pred = GefFitter._linear_model(self.time, slope, F0)

            elif model == 'exponential_model':
                y_pred = GefFitter._exponential_model(self.time,  span_exchange, k_exchange, fluorescence_plateau)

            elif model == 'exponential_model_with_background':
                y_pred = GefFitter._exponential_model_with_background(self.time, span_exchange, k_exchange, span_background, k_background, fluorescence_plateau)

            colors.append(color)
            labels.append(label)
            ydata.append(progress_curve)

            plot_data.append({
                'title': title,
                'colors': [color],
                'ypreds': [y_pred],
                'labels': [label],
                'ydata': [progress_curve]
            })

        plot_data.insert(0, {
            'title': 'Progress Curves',
            'colors': colors,
            'ypreds': [],
            'labels': labels,
            'ydata': ydata
        })

        # launch an interactive figure
        self._launch_interactive_figure(plot_data, xlabel, ylabel, width_per_plot, height_per_plot)

    def _launch_interactive_figure(self, plot_data, xlabel, ylabel, width_per_plot, height_per_plot):
        
        sns.set_style('ticks')
        prev_button = widgets.Button(description="Previous")
        next_button = widgets.Button(description="Next")

        # Output area to display the plots
        output = widgets.Output()

        def display_plot(index):
            with output:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(width_per_plot, height_per_plot))

                if index == 0:
                    ax.set_title(plot_data[index]['title'])
                    for progress_curve, label, color,  in zip(plot_data[index]['ydata'], plot_data[index]['labels'], plot_data[index]['colors']):
                        ax.plot(self.time, progress_curve, label=label, color=color)

                else:
                    ax.set_title(plot_data[index]['title'])
                    ax.plot(self.time, plot_data[index]['ydata'][0], color=plot_data[index]['colors'][0])
                    ax.plot(self.time, plot_data[index]['ypreds'][0], color='black')

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                plt.show()

        def on_prev_button_clicked(b):
            if self.current_index > 0:
                self.current_index -= 1
            display_plot(self.current_index)

        def on_next_button_clicked(b):
            if self.current_index < len(plot_data) - 1:
                self.current_index += 1
            display_plot(self.current_index)

        # Attach button click event handlers
        prev_button.on_click(on_prev_button_clicked)
        next_button.on_click(on_next_button_clicked)

        # Display widgets and initial plot
        display(widgets.HBox([prev_button, next_button], layout={'justify_content': 'center'}), output)
        display_plot(self.current_index)

    def fit_initial_rates(self, 
                          initial_guess_and_constraints=False,
                          plot=False, 
                          xlabel="Time (s)",
                          ylabel="Trp Fluorescence (RFUs)",
                          height_per_plot=10,
                          width_per_plot=10,
                          palette:list=None):
        
        # parse through input data to obtain substrate and enzyme concentrations
        slopes, fluorescence_plateaus, k_exchanges, k_backgrounds, span_exchanges, span_backgrounds, pconvs, vF0s, F0s = [], [], [], [], [], [], [], [], []
        for header, ydata in zip(self.headers, self.ydatas):
            conc, _, fit_type, perc_curve, _ = self._get_fit_inputs(header)

            if fit_type == 'linear_model':

                if type(perc_curve) == float:

                    # truncate x and y data
                    linear_regime_end = round(len(self.time) * perc_curve)
                    time_trunc = self.time[0:linear_regime_end] 
                    ydata_trunc = ydata[0:linear_regime_end]  

                    # fit linear model
                    bounds = Bounds(lb=np.array([0, 0]), ub=np.array([np.inf, np.inf]))
                    popt, pconv = curve_fit(GefFitter._linear_model, time_trunc, ydata_trunc, bounds=(bounds.lb, bounds.ub))
                    slope, yint = popt

                else:
                    slopes_out, yint, r2 = compute_initial_reaction_slope(self.time, ydata, min_included_percent=2)
                    # print(slopes, yint)
                    slope = slopes_out[0] * -1
                    pconv = 'NA'
                
                # organize parameters
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
            self._plot_progress_curves_and_fits(xlabel, ylabel, palette, width_per_plot, height_per_plot)

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
            self._plot_progress_curves_and_fits(xlabel, ylabel, palette, width_per_plot, height_per_plot)

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
            x = np.linspace(0, max(substrate_concens) * 2, 1000)
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

    def generate_sse_surfaces_MM(self, 
                                 plot:bool=False, 
                                 image_path:str=None, 
                                 n_points:int=30, 
                                 height_per_plot=10, 
                                 width_per_plot=10,):
        
        initial_velocities = self.data_index['vF0'].to_numpy() * self.conversion_factor_fit.slope # convert to molar concentration
        substrate_concens = self.data_index['conc'].to_numpy() 
        gef_concs = list(set(self.data_index['GEF_conc']))
        gef_conc = gef_concs.pop()

        # compute surface ranging over 2-fold below and above the Km
        Km_array = np.linspace(self.Km / 2, self.Km * 2, n_points)
        Km_sse_squared_array = []
        for Km in Km_array:
            wrapper = lambda substrate_concens, Vmax: GefFitter._michaelis_menten_model(substrate_concens, Vmax, Km)
            Vmax, _ = curve_fit(wrapper, substrate_concens, initial_velocities)
            y_hat = GefFitter._michaelis_menten_model(substrate_concens, Vmax, Km)
            sse = np.square(y_hat - initial_velocities).sum()
            Km_sse_squared_array.append(sse)
            
        # compute surface ranging over 2-fold below and above the kcat
        kcat_array = np.linspace(self.kcat / 2, self.kcat * 2, n_points)
        kcat_sse_squared_array = []
        for kcat in kcat_array:
            wrapper = lambda substrate_concens, Km: GefFitter._michaelis_menten_model(substrate_concens, kcat * (gef_conc / 1000), Km)
            Km, _ = curve_fit(wrapper, substrate_concens, initial_velocities)
            y_hat = GefFitter._michaelis_menten_model(substrate_concens, kcat * (gef_conc / 1000), Km)
            sse = np.square(y_hat - initial_velocities).sum()
            kcat_sse_squared_array.append(sse)
                    
        self.Km_sse_squared_array = np.array(Km_sse_squared_array)
        self.Km_array = Km_array
        self.kcat_sse_squared_array = np.array(kcat_sse_squared_array)
        self.kcat_array = kcat_array

        if plot:

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(width_per_plot * 2, height_per_plot))
            axs[0].set_title(r'$K_{m}$ SSE Surface')
            axs[0].set_xlabel(r'$K_{m}$' + ' (µM)')
            axs[0].set_ylabel('SSE')
            axs[1].set_title(r'$k_{cat}$ SSE Surface')
            axs[1].set_xlabel(r'$k_{cat}$' + ' $s^{-1}$')
            axs[1].set_ylabel('SSE')
            axs[0].scatter(self.Km_array, self.Km_sse_squared_array, color='black')
            axs[1].scatter(self.kcat_array, self.kcat_sse_squared_array, color='black')

            if image_path:
                plt.savefig(os.path.join(image_path), dpi=300)

