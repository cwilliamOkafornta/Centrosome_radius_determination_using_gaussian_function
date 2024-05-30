# library
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, linspace
from scipy.optimize import curve_fit

# create a class for the computation function
class MTGaussianFitting:
    def __init__(self, folder_input, file_input, csvfilename, save_folder, csvsavefilename):
        self.folder_input = folder_input
        self.file_input = file_input
        self.csvfilename = csvfilename
        self.save_folder = save_folder
        self.csvsavefilename = csvsavefilename
        self.data_file = self.fileLoader()

    '''load the input file(s)'''
    def fileLoader(self):
        data_file = []
        for file in self.file_input:
            filepath = os.path.join(self.folder_input, file)
            filename = os.path.basename(filepath)
            if filename == self.csvfilename:
                try:
                    df = pd.read_csv(filepath, index_col=None, encoding='utf-8')
                except:
                    df = pd.read_csv(filepath, index_col=None, encoding='cp1252')
                data_file.append(df)
        return data_file

    '''extract the columns to be analysed from the DataFrame'''
    def poleMTsDF(self, columnname, Pole_MT_id1, Pole_MT_id2):
        df_table = self.data_file[0]
        df_table_pole = df_table[df_table['Fiber_Name'].str.contains(f'{Pole_MT_id1}|{Pole_MT_id2}')][[columnname]]
        df_pole = df_table_pole.reset_index(drop=True)
        return df_pole

    '''define the gaussian function'''
    @staticmethod
    def gaussian(x, A, x0, sigma):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

    '''fit the gaussian function on the data histogram'''
    def fitGaussFunc(self, MTs_df):
        counts, bin_edges = np.histogram(MTs_df, bins='fd', density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Convert the DataFrame column to a NumPy array
        MTs_array = MTs_df.to_numpy().flatten()
    
        mean_data = np.mean(MTs_array)
        std_data = np.std(MTs_array)  # No need to specify axis, as it's now a 1D array
        amplitude_value = np.max(counts)
        initial_guess_values = [amplitude_value, mean_data, std_data]
    
        popt, _ = curve_fit(self.gaussian, bin_centers, counts, p0=initial_guess_values)
        A, x0, sigma = popt
    
        x_fit = np.linspace(np.min(MTs_array), np.max(MTs_array), 1000)
        y_fit = self.gaussian(x_fit, *popt)
        return A, x0, sigma, x_fit, y_fit, counts, bin_edges, bin_centers

    '''determine the half-maximun value of the fitted curve and extract the corresponding values on x-axis'''
    @staticmethod
    def half_max(y_fit, x_fit, A):
        half_maximum = A / 2
        number_close_to_zero = np.abs(y_fit - half_maximum)
        first_index_close_to_zero = np.argmin(number_close_to_zero)
        index_maximum_value = np.argmax(number_close_to_zero)
        value_maximum_value = number_close_to_zero[index_maximum_value]
        number_close_to_zero[first_index_close_to_zero] = value_maximum_value
        second_index_close_to_zero = np.argmin(number_close_to_zero)
        new_x_intercept_half_maximum = [x_fit[first_index_close_to_zero], x_fit[second_index_close_to_zero]]
        x2_fit_value = np.max(new_x_intercept_half_maximum)
        return x2_fit_value, half_maximum

    '''extract the measured parameter from the previous functions and save as a new DataFrame'''
    def parameterTable(self, pole1_params, pole2_params):
        parameters_values_pole1 = pd.DataFrame.from_dict({
            'Maximum peak': pole1_params[0],
            'Mean center of peak': pole1_params[1],
            'Width of the peak': pole1_params[2],
            'Radius of the centrosome': pole1_params[3],
        }, orient='index', columns=['Pole1 (µm)'])

        parameters_values_pole2 = pd.DataFrame.from_dict({
            'Maximum peak': pole2_params[0],
            'Mean center of peak': pole2_params[1],
            'Width of the peak': pole2_params[2],
            'Radius of the centrosome': pole2_params[3],
        }, orient='index', columns=['Pole 2 (µm)'])

        parameter_df = pd.concat([parameters_values_pole1, parameters_values_pole2], axis=1, ignore_index=False)
        parameter_df.index.names = ['Parameters']
        parameter_df.to_csv(os.path.join(self.save_folder, self.csvsavefilename + '.csv'), index=True, encoding='cp1252')

    '''visualize the plot and save'''
    def plotHistogram(self, pole1_params, pole2_params, plot_params):
        pole1_x_fit, pole1_y_fit, pole1_counts, pole1_bin_edges, pole1_bin_centers = pole1_params[3:8]
        pole2_x_fit, pole2_y_fit, pole2_counts, pole2_bin_edges, pole2_bin_centers = pole2_params[3:8]
        x2_fit_value_pole1, pole1_half_maximum = pole1_params[8:10]
        x2_fit_value_pole2, pole2_half_maximum = pole2_params[8:10]

        plt.figure(figsize=(6, 6))
        plt.bar(pole1_bin_centers, pole1_counts, width=pole1_bin_edges[1] - pole1_bin_edges[0], color=plot_params['pole1_hist_color'], edgecolor='k', label=plot_params['pole1_label'], alpha=0.6)
        plt.bar(pole2_bin_centers, pole2_counts, width=pole2_bin_edges[1] - pole2_bin_edges[0], color=plot_params['pole2_hist_color'], edgecolor='k', label=plot_params['pole2_label'], alpha=0.4)
        plt.plot(pole1_x_fit, pole1_y_fit, linewidth=3, label=plot_params['pole1_fit_label'], color=plot_params['pole1_fit_line_color'])
        plt.plot(pole2_x_fit, pole2_y_fit, linewidth=3, label=plot_params['pole2_fit_label'], color=plot_params['pole2_fit_line_color'])
        plt.vlines(x2_fit_value_pole1, 0, pole1_half_maximum, colors=plot_params['pole1_vline_color'], linestyles='dashed')
        plt.hlines(pole1_half_maximum, 0, x2_fit_value_pole1, colors=plot_params['pole1_hline_color'], linestyles='dashed')
        plt.vlines(x2_fit_value_pole2, 0, pole2_half_maximum, colors=plot_params['pole2_vline_color'], linestyles='dashed')
        plt.hlines(pole2_half_maximum, 0, x2_fit_value_pole2, colors=plot_params['pole2_hline_color'], linestyles='dashed')
        plt.xlim(plot_params['minimum_xlim'], plot_params['maximum_xlim'])
        plt.ylim(plot_params['minimum_ylim'], plot_params['maximum_ylim'])
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='x', direction='in')
        plt.tick_params(axis='y', direction='in')
        plt.title(plot_params['plot_title'], fontsize=plot_params['plot_title_fontsize'])
        plt.xlabel(plot_params['xaxis_title'], fontsize=plot_params['xaxis_label_fontsize'])
        plt.ylabel(plot_params['yaxis_title'], fontsize=plot_params['yaxis_label_fontsize'])
        plt.legend()

        plt.savefig(os.path.join(self.save_folder, plot_params['plot_filename'] + '.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.save_folder, plot_params['plot_filename'] + '.svg'), dpi=300, bbox_inches='tight')

    '''call all the functions within the same class'''
    def GaussianFitHistoPlot(self, columnname1, Pole1_MT_id1, Pole1_MT_id2, columnname2, Pole2_MT_id1, Pole2_MT_id2, plot_params):
        Pole1_MTs_df = self.poleMTsDF(columnname1, Pole1_MT_id1, Pole1_MT_id2)
        Pole2_MTs_df = self.poleMTsDF(columnname2, Pole2_MT_id1, Pole2_MT_id2)

        pole1_params = self.fitGaussFunc(Pole1_MTs_df)
        pole2_params = self.fitGaussFunc(Pole2_MTs_df)

        x2_fit_value_pole1, pole1_half_maximum = self.half_max(pole1_params[4], pole1_params[3], pole1_params[0])
        x2_fit_value_pole2, pole2_half_maximum = self.half_max(pole2_params[4], pole2_params[3], pole2_params[0])

        pole1_params += (x2_fit_value_pole1, pole1_half_maximum)
        pole2_params += (x2_fit_value_pole2, pole2_half_maximum)

        self.parameterTable(pole1_params, pole2_params)
        self.plotHistogram(pole1_params, pole2_params, plot_params)