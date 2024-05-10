# library
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, linspace, random, arange
from scipy.optimize import curve_fit, least_squares

def fileLoader(folder_input, file_input, csvfilename):
    # read out the file of interest and include into a list
    data_file = []
    for file in file_input:
        filepath = os.path.join(folder_input, file)
        filename = os.path.basename(filepath)
        if filename == csvfilename:
            try:
                df = pd.read_csv(filepath, index_col=None, encoding='utf-8')
            except:
                df = pd.read_csv(filepath, index_col=None, encoding='cp1252')
            data_file.append(df)
    return data_file

'''make a new DataFrame for the MTs minus end distance to pole 1'''
def poleOneMTsDF(filetoanalyze, columnname1, Pole1_MT_id1, Pole1_MT_id2):
    df_table = filetoanalyze[0]
    df_table_pole1 = df_table[df_table['Fiber_Name'].str.contains(f'{Pole1_MT_id1}|{Pole1_MT_id2}')][[columnname1]]
    df_pole1 = df_table_pole1.reset_index(drop=True)
    return df_pole1

'''make a new DataFrame for the MTs minus end distance to pole 2'''
def poleTwoMTsDF(filetoanalyze, columnname2, Pole2_MT_id1, Pole2_MT_id2):
    df_table = filetoanalyze[0]
    df_table_pole2 = df_table[df_table['Fiber_Name'].str.contains(f'{Pole2_MT_id1}|{Pole2_MT_id2}')][[columnname2]]
    df_pole2 = df_table_pole2.reset_index(drop=True)
    return df_pole2

# gaussian fit function
def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

'''Fit the gaussian function on the MTs of spindle pole 1'''
def fitGuassFuncPoleOne(Pole1_MTs_df):
    # compute the histogram for the MTs data of pole 1
    counts_p1, bin_edges_p1 = np.histogram(Pole1_MTs_df, bins='fd', density=True) # bins size adopts the Freedman Diaconis Estimator
    bin_centers_p1 = (bin_edges_p1[:-1] + bin_edges_p1[1:]) / 2 # determine the center of each bin of the histogram

    # initial guess values for the guassian parameters; A, x0, and sigma
    mean_data_pole1 = np.mean(Pole1_MTs_df)
    std_data_pole1 = np.std(Pole1_MTs_df)
    _, std_pole1_factorize = pd.factorize(std_data_pole1)
    df_std_pole1 = pd.DataFrame(std_pole1_factorize, columns=['minus_dist_to_pole[Pole1_01 | Pole1_02]'])
    std_pole1_values = float(df_std_pole1['minus_dist_to_pole[Pole1_01 | Pole1_02]'].to_string(index=False))
    amplitude_value = np.max(counts_p1)
    initial_guess_values_p1 = [amplitude_value, mean_data_pole1, std_pole1_values] 

    # fit gaussian model for pole1
    popt_p1, pcov_p1 = curve_fit(gaussian, bin_centers_p1, counts_p1, p0=initial_guess_values_p1)
    A_p1, x0_p1, sigma_p1 = popt_p1

    # x_fit and y_fit values 
    x_fit_p1 = np.linspace(np.min(Pole1_MTs_df), np.max(Pole1_MTs_df), 1000)
    y_fit_p1 = gaussian(x_fit_p1, *popt_p1)
    return A_p1, x0_p1, sigma_p1, x_fit_p1, y_fit_p1, counts_p1, bin_edges_p1, bin_centers_p1

'''Fit the gaussian function on the MTs of spindle pole 2'''
def fitGuassFuncPoleTwo(Pole2_MTs_df):
    # compute the histogram for the MTs data of pole 2
    counts_p2, bin_edges_p2 = np.histogram(Pole2_MTs_df, bins='fd', density=True) # bins size adopts the Freedman Diaconis Estimator
    bin_centers_p2 = (bin_edges_p2[:-1] + bin_edges_p2[1:]) / 2 # determine the center of each bin of the histogram

    # initial guess values for the guassian parameters; A, x0, and sigma
    mean_data_Pole2 = np.mean(Pole2_MTs_df)
    std_data_Pole2 = np.std(Pole2_MTs_df)
    _, std_Pole2_factorize = pd.factorize(std_data_Pole2)
    df_std_Pole2 = pd.DataFrame(std_Pole2_factorize, columns=['minus_dist_to_pole[Pole2_01 | Pole2_02]'])
    std_Pole2_values = float(df_std_Pole2['minus_dist_to_pole[Pole2_01 | Pole2_02]'].to_string(index=False))
    amplitude_value = np.max(counts_p2)
    initial_guess_values_p2 = [amplitude_value, mean_data_Pole2, std_Pole2_values] 

    # fit gaussian model for Pole2
    popt_p2, pcov_p2 = curve_fit(gaussian, bin_centers_p2, counts_p2, p0=initial_guess_values_p2)
    A_p2, x0_p2, sigma_p2 = popt_p2

    # x_fit and y_fit values 
    x_fit_p2 = np.linspace(np.min(Pole2_MTs_df), np.max(Pole2_MTs_df), 1000)
    y_fit_p2 = gaussian(x_fit_p2, *popt_p2)
    return A_p2, x0_p2, sigma_p2, x_fit_p2, y_fit_p2, counts_p2, bin_edges_p2, bin_centers_p2

# '''Calculate for the Full width of the peak at half maximum (FWHM)'''
# def FWHM(pole1_peak_center, pole1_peak_width, pole2_peak_center, pole2_peak_width):
#     sqrt_2ln2 = np.sqrt(2 * np.log(2)) # standard calculation
    
#     # pole 1 
#     x1_fit_value_pole1 = pole1_peak_center - pole1_peak_width * sqrt_2ln2
#     x2_fit_value_pole1 = pole1_peak_center + pole1_peak_width * sqrt_2ln2
    
#     # pole 2
#     x1_fit_value_pole2 = pole2_peak_center - pole2_peak_width * sqrt_2ln2
#     x2_fit_value_pole2 = pole2_peak_center + pole2_peak_width * sqrt_2ln2
#     return x1_fit_value_pole1, x2_fit_value_pole1, x1_fit_value_pole2, x2_fit_value_pole2

'''calculate for the Full width of the peak at half maximum (FWHM)'''
def half_max(y_fit_p1, x_fit_p1, A_p1, y_fit_p2, x_fit_p2, A_p2):
    '''Pole 1'''
    pole1_half_maximum = A_p1/2                                                                 # half maximum peak value of pole 1
    pole1_number_close_to_zero = np.abs(y_fit_p1 - pole1_half_maximum)                          # absolute difference of pole 1 y-fit value to find the point closest to the half-maximum peak value 
    pole1_first_index_close_to_zero = np.argmin(pole1_number_close_to_zero)                     # index of the minimum value closest to the half-maximum peak to find the edge of the peak
    pole1_index_maximum_value = np.argmax(pole1_number_close_to_zero)                           # index of the maximum value further away form the half-maximum peak
    pole1_value_maximum_value = pole1_number_close_to_zero[pole1_index_maximum_value]           # maximum value further away from the half-maximum peak
    pole1_number_close_to_zero[pole1_first_index_close_to_zero] = pole1_value_maximum_value     # replace the first index value with the maximum value
    pole1_second_index_close_to_zero = np.argmin(pole1_number_close_to_zero)                    # index of the second minimum value closest to the half-maximum peak to find the edge of the peak
    new_x_intercept_half_maximum = [x_fit_p1[pole1_first_index_close_to_zero], x_fit_p1[pole1_second_index_close_to_zero]]  # x-intercepts of the half-maximum peak
    x2_fit_value_pole1 = np.max(new_x_intercept_half_maximum)                                   # x2 value of the half-maximum peak
    
    '''Pole 2'''
    pole2_half_maximum = A_p2/2                                                                 # half maximum peak value of Pole 2
    pole2_number_close_to_zero = np.abs(y_fit_p2 - pole2_half_maximum)                          # absolute difference of Pole 2 y-fit value to find the point closest to the half-maximum peak value 
    pole2_first_index_close_to_zero = np.argmin(pole2_number_close_to_zero)                     # index of the minimum value closest to the half-maximum peak to find the edge of the peak
    pole2_index_maximum_value = np.argmax(pole2_number_close_to_zero)                           # index of the maximum value further away form the half-maximum peak
    pole2_value_maximum_value = pole2_number_close_to_zero[pole2_index_maximum_value]           # maximum value further away from the half-maximum peak
    pole2_number_close_to_zero[pole2_first_index_close_to_zero] = pole2_value_maximum_value     # replace the first index value with the maximum value
    pole2_second_index_close_to_zero = np.argmin(pole2_number_close_to_zero)                    # index of the second minimum value closest to the half-maximum peak to find the edge of the peak
    new_x_intercept_half_maximum = [x_fit_p2[pole2_first_index_close_to_zero], x_fit_p2[pole2_second_index_close_to_zero]]  # x-intercepts of the half-maximum peak
    x2_fit_value_pole2 = np.max(new_x_intercept_half_maximum)                                   # x2 value of the half-maximum peak
    
    return x2_fit_value_pole1, x2_fit_value_pole2, pole1_half_maximum, pole2_half_maximum

'''create a dataframe for calculated parameters of pole 1 and pole 2'''
def parameterTable(save_folder, csvsavefilename, pole1_max_amplitude, pole1_peak_center, pole1_peak_width, x2_fit_value_pole1,
                   pole2_max_amplitude, pole2_peak_center, pole2_peak_width, x2_fit_value_pole2):
    parameters_values_pole1 = pd.DataFrame.from_dict({'Maximum peak': pole1_max_amplitude, 
                                                      'Mean center of peak': pole1_peak_center, 
                                                      'Width of the peak': pole1_peak_width, 
                                                      'Radius of the centrosome': x2_fit_value_pole1 ,
                                                     }, orient='index', columns=['Pole1 (µm)'])

    parameters_values_pole2 = pd.DataFrame.from_dict({'Maximum peak': pole2_max_amplitude, 
                                                      'Mean center of peak': pole2_peak_center, 
                                                      'Width of the peak': pole2_peak_width, 
                                                      'Radius of the centrosome': x2_fit_value_pole2,
                                                     }, orient='index', columns=['Pole 2 (µm)'])

    parameter_df = pd.concat([parameters_values_pole1, parameters_values_pole2], axis=1, ignore_index=False)
    parameter_df.index.names = ['Parameters']
    parameter_df.to_csv(os.path.join(save_folder, csvsavefilename+'.csv'), index=True, encoding='cp1252')
    
def plotHistogram(pole1_x_fit, pole1_y_fit, pole1_counts, pole1_bin_edges, pole1_bin_centers,
                  pole2_x_fit, pole2_y_fit, pole2_counts, pole2_bin_edges, pole2_bin_centers,
                  x2_fit_value_pole1, x2_fit_value_pole2, pole1_half_maximum, pole2_half_maximum,
                  pole1_hist_color, pole1_label, pole1_fit_label, pole1_fit_line_color, pole1_vline_color, pole1_hline_color, 
                  pole2_hist_color, pole2_label, pole2_fit_label, pole2_fit_line_color, pole2_vline_color, pole2_hline_color,
                  minimum_xlim, maximum_xlim, 
                  minimum_ylim, maximum_ylim, 
                  plot_title_fontsize, xaxis_label_fontsize, 
                  yaxis_label_fontsize, plot_title, xaxis_title, yaxis_title,
                  save_folder, plot_filename):
    
    # Plot the histogram and the fit
    plt.figure(figsize=(6, 6))
    plt.bar(pole1_bin_centers, pole1_counts, width=pole1_bin_edges[1] - pole1_bin_edges[0], color=pole1_hist_color, edgecolor='k', label=pole1_label)
    plt.bar(pole2_bin_centers, pole2_counts, width=pole2_bin_edges[1] - pole2_bin_edges[0], color=pole2_hist_color, edgecolor='k', label=pole2_label)
    plt.plot(pole1_x_fit, pole1_y_fit, linewidth=3, label=pole1_fit_label, color=pole1_fit_line_color)
    plt.plot(pole2_x_fit, pole2_y_fit, linewidth=3, label=pole2_fit_label, color=pole2_fit_line_color)
    plt.vlines(x2_fit_value_pole1, 0, pole1_half_maximum, colors=pole1_vline_color, linestyles='dashed')
    plt.hlines(pole1_half_maximum, 0, x2_fit_value_pole1, colors=pole1_hline_color, linestyles='dashed')
    plt.vlines(x2_fit_value_pole2 , 0, pole2_half_maximum, colors=pole2_vline_color, linestyles='dashed')
    plt.hlines(pole2_half_maximum, 0, x2_fit_value_pole2 , colors=pole2_hline_color, linestyles='dashed')
    plt.xlim(minimum_xlim, maximum_xlim)
    plt.ylim(minimum_ylim, maximum_ylim)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.title(plot_title, fontsize=plot_title_fontsize)
    plt.xlabel(xaxis_title, fontsize=xaxis_label_fontsize)
    plt.ylabel(yaxis_title, fontsize=yaxis_label_fontsize)
    plt.legend()

    # save the plot
    plt.savefig(os.path.join(save_folder, plot_filename+'.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, plot_filename+'.svg'), dpi=300, bbox_inches='tight')
    
def GuassianFitHistoPlot(folder_input, 
                         file_input, 
                         csvfilename, 
                         save_folder, 
                         csvsavefilename,
                         columnname1, 
                         Pole1_MT_id1, 
                         Pole1_MT_id2, 
                         columnname2, 
                         Pole2_MT_id1, 
                         Pole2_MT_id2, 
                         pole1_hist_color, 
                         pole1_label, 
                         pole1_fit_label, 
                         pole1_fit_line_color, 
                         pole1_vline_color, 
                         pole1_hline_color, 
                         pole2_hist_color, 
                         pole2_label, 
                         pole2_fit_label, 
                         pole2_fit_line_color, 
                         pole2_vline_color, 
                         pole2_hline_color,
                         minimum_xlim, 
                         maximum_xlim,
                         minimum_ylim, 
                         maximum_ylim, 
                         plot_title_fontsize, 
                         xaxis_label_fontsize, 
                         yaxis_label_fontsize, 
                         plot_title, 
                         xaxis_title, 
                         yaxis_title, 
                         plot_filename):
    
    filetoanalyze = fileLoader(folder_input, file_input, csvfilename,) 
    Pole1_MTs_df = poleOneMTsDF(filetoanalyze, columnname1, Pole1_MT_id1, Pole1_MT_id2) # pole dataframe 
    Pole2_MTs_df = poleTwoMTsDF(filetoanalyze, columnname2, Pole2_MT_id1, Pole2_MT_id2) # pole 2 dataframe 
    pole1_max_amplitude = fitGuassFuncPoleOne(Pole1_MTs_df)[0] 
    pole1_peak_center = fitGuassFuncPoleOne(Pole1_MTs_df)[1] 
    pole1_peak_width = fitGuassFuncPoleOne(Pole1_MTs_df)[2] 
    pole1_x_fit = fitGuassFuncPoleOne(Pole1_MTs_df)[3] 
    pole1_y_fit = fitGuassFuncPoleOne(Pole1_MTs_df)[4] 
    pole1_counts = fitGuassFuncPoleOne(Pole1_MTs_df)[5] 
    pole1_bin_edges = fitGuassFuncPoleOne(Pole1_MTs_df)[6] 
    pole1_bin_centers = fitGuassFuncPoleOne(Pole1_MTs_df)[7] 
    pole2_max_amplitude = fitGuassFuncPoleTwo(Pole2_MTs_df)[0] 
    pole2_peak_center = fitGuassFuncPoleTwo(Pole2_MTs_df)[1] 
    pole2_peak_width = fitGuassFuncPoleTwo(Pole2_MTs_df)[2] 
    pole2_x_fit = fitGuassFuncPoleTwo(Pole2_MTs_df)[3] 
    pole2_y_fit = fitGuassFuncPoleTwo(Pole2_MTs_df)[4] 
    pole2_counts = fitGuassFuncPoleTwo(Pole2_MTs_df)[5] 
    pole2_bin_edges = fitGuassFuncPoleTwo(Pole2_MTs_df)[6] 
    pole2_bin_centers = fitGuassFuncPoleTwo(Pole2_MTs_df)[7] 
    x2_fit_value_pole1, x2_fit_value_pole2, pole1_half_maximum, pole2_half_maximum = half_max(pole1_y_fit, pole1_x_fit, pole1_max_amplitude, pole2_y_fit, pole2_x_fit, pole2_max_amplitude)
    # x1_fit_value_pole1, x2_fit_value_pole1, x1_fit_value_pole2, x2_fit_value_pole2 = FWHM(pole1_peak_center, pole1_peak_width, pole2_peak_center, pole2_peak_width)

    parameterTable(save_folder, csvsavefilename, pole1_max_amplitude, pole1_peak_center, pole1_peak_width, x2_fit_value_pole1,
                   pole2_max_amplitude, pole2_peak_center, pole2_peak_width, x2_fit_value_pole2)
    plotHistogram(pole1_x_fit, pole1_y_fit, pole1_counts, pole1_bin_edges, pole1_bin_centers,
                  pole2_x_fit, pole2_y_fit, pole2_counts, pole2_bin_edges, pole2_bin_centers,
                  x2_fit_value_pole1, x2_fit_value_pole2, pole1_half_maximum, pole2_half_maximum,
                  pole1_hist_color, pole1_label, pole1_fit_label, pole1_fit_line_color, pole1_vline_color, pole1_hline_color, 
                  pole2_hist_color, pole2_label, pole2_fit_label, pole2_fit_line_color, pole2_vline_color, pole2_hline_color,
                  minimum_xlim, maximum_xlim, 
                  minimum_ylim, maximum_ylim, 
                  plot_title_fontsize, xaxis_label_fontsize, 
                  yaxis_label_fontsize, plot_title, xaxis_title, yaxis_title,
                  save_folder, plot_filename)