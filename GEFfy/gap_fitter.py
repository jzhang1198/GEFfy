import pandas as pd
import numpy as np
import seaborn as sns
import ipywidgets as widgets
from scipy.special import lambertw
from matplotlib import pyplot as plt
from IPython.display import display, clear_output

class GapFitter:
    def __init__(self, data: pd.DataFrame, data_index: pd.DataFrame, concentration_units:str='µM', time_units:str='s'):
        
        # set units
        self.concentration_units = concentration_units
        self.time_units = time_units

        # get data
        self.headers = data.columns[1:].to_list()
        self.time = data.to_numpy(dtype=np.float64).T[0]
        self.ydatas = data.to_numpy(dtype=np.float64).T[1:]
        self.data_index = data_index # dynamically updated with fit parameters
        self.current_index= 0 # attribute for interactive plotting

        # checks for data quality
        if len(set(data_index['sample'].to_list())) != len(data_index['sample'].to_list()):
            print('WARNING: duplicate sample IDs detected. Please ensure that all samples have a unique identifier.')
            print('Exited initialization.')
            return
        
        # initialize empty attributes to be assigned during fitting
        self.Km = float
        self.kcat = float

        self.Km_sse_squared_array = np.ndarray
        self.Km_array = np.ndarray
        self.kcat_sse_squared_array = np.ndarray
        self.kcat_array = np.ndarray

    def _get_fit_inputs(self, sample_id: str):
        row = self.data_index[self.data_index['sample'] == sample_id]
        conc, GAP_conc, sensor_conc, date, model = row.iloc[0]['conc'], row.iloc[0]['GAP_conc'], row.iloc[0]['sensor_conc'], row.iloc[0]['date'], row.iloc[0]['model']
        return conc, GAP_conc, sensor_conc, date, model
    
    def _get_fit_outputs(self, sample_id: str):
        row = self.data_index[self.data_index['sample'] == sample_id]
        F0, FF, Km, kcat, beta, pconv = row.iloc[0]['F0'], row.iloc[0]['FF'], row.iloc[0]['Km'], row.iloc[0]['kcat'], row.iloc[0]['beta'], row.iloc[0]['pconv']  
        return F0, FF, Km, kcat, beta, pconv

    @staticmethod
    def _integrated_michaelis_menten_equation(time: np.ndarray, Km: float, Vmax: float, s0: float):
        z = (s0 / Km) * np.exp(np.subtract(s0, Vmax * time) / Km)
        product_concens = s0 - (Km * np.real(lambertw(z))) # assuming we only have to use the principal real branch
        return product_concens
    
    @staticmethod
    def _fitting_function_background(time: np.ndarray, beta: float):
        return beta * time

    @staticmethod
    def _fitting_function_signal(time: np.ndarray, F0: float, FF: float, Km:float, Vmax: float, s0: float):
        m = (FF - F0) / s0
        fluorescence = m * GapFitter._integrated_michaelis_menten_equation(time, Km, Vmax, s0) + F0
        return fluorescence    
    
    def _plot_progress_curves_and_fits(self, xlabel: str, ylabel: str, width_per_plot: float, height_per_plot: float):
        
        # add units to xlabel
        xlabel = xlabel + f' ({self.time_units})'

        plot_data = []
        for index, header in enumerate(self.headers):
            conc, GAP_conc, _, _, model = self._get_fit_inputs(header)
            F0, FF, Km, kcat, beta, _ = self._get_fit_outputs(header)
            progress_curve = self.ydatas[index]

            kcat_subtitle = '$k_{cat}$ = ' + '{:.2f}'.format(kcat) + f'${self.time_units}^{-1}$'
            Km_subtitle = ', $K_m$ = ' + '{:.2f}'.format(Km) + self.concentration_units
            title = kcat_subtitle + Km_subtitle

            if model == 'integrated_MM':
                y_pred = GapFitter._fitting_function_signal(self.time, F0, FF, Km, GAP_conc * kcat, conc)

            elif model == 'integrated_MM_with_background':
                y_pred = GapFitter._fitting_function_signal(self.time, F0, FF, Km, GAP_conc * kcat, conc) + GapFitter._fitting_function_background(self.time, beta)

            plot_data.append(
                {'title': title,
                 'ydata': progress_curve,
                 'ypred': y_pred}
            )

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
                ax.set_title(plot_data[index]['title'])
                ax.plot(self.time, plot_data[index]['ydata'], color='black')
                ax.plot(self.time, plot_data[index]['ypred'], color='red')
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

    def fit_integrated_MM(self, 
                          plot:bool=False, 
                          image_path:str=None, 
                          title:str='GAP-catalyzed Hydrolysis Integrated MM Fit',
                          xlabel:str='Time',
                          ylabel:str='PBP Fluorescence (RFUs)',
                          width_per_plot:float=8,
                          height_per_plot:float=8):

        F0s, FFs, Kms, kcats, betas, pconvs = [], [], [], [], [], []
        for header, ydata in zip(self.headers, self.ydatas):
            substrate_conc, GAP_conc, _, _, model = self._get_fit_inputs(header)            

            if model == 'integrated_MM':
                wrapper = lambda time, F0, FF, Km, Vmax: GapFitter._fitting_function_signal(time, F0, FF, Km, Vmax, substrate_conc)
                lb, ub = np.array([0, 0, 0, 0]), np.array([np.inf, np.inf, np.inf, np.inf])
                initial_guess = np.array([ydata[0], ydata[-1], 1, 0.001])
                popt, pconv = curve_fit(wrapper, self.time, ydata, bounds=(lb, ub), p0=initial_guess)

                # unpack and organize
                F0, FF, Km, Vmax = popt 
                F0s.append(F0)
                FFs.append(FF)
                Kms.append(Km)
                kcats.append(Vmax / (GAP_conc))
                betas.append('NA')
                pconvs.append(pconv)

            elif model == 'integrated_MM_with_background':
                wrapper = lambda time, F0, FF, Km, Vmax, beta: GapFitter._fitting_function_signal(time, F0, FF, Km, Vmax, substrate_conc) + GapFitter._fitting_function_background(time, beta)
                lb, ub = np.array([0, 0, 0, 0, -np.inf]), np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
                initial_guess = np.array([ydata[0], ydata[-1], 1, 0.001, 1])
                popt, pconv = curve_fit(wrapper, self.time, ydata, bounds=(lb, ub), p0=initial_guess)

                # unpack and organize
                F0, FF, Km, Vmax, beta = popt 
                F0s.append(F0)
                FFs.append(FF)
                Kms.append(Km)
                kcats.append(Vmax / (GAP_conc))
                betas.append(beta)
                pconvs.append(pconv)

            else:
                print(f'ERROR: f{model} not recognized as a valid model to fit to. Valid models include "integrated_MM" or "integrated_MM_with_background".')
                return
        
        self.data_index['F0'] = F0s
        self.data_index['FF'] = FFs
        self.data_index['Km'] = Kms 
        self.data_index['kcat'] = kcats
        self.data_index['beta'] = betas
        self.data_index['pconv'] = pconvs

        if plot:
            self._plot_progress_curves_and_fits(xlabel, ylabel, width_per_plot, height_per_plot)