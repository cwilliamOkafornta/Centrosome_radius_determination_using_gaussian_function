# Centrosome_radius_determination_using_gaussian_function
Centrosome radius determination from a typical mitotic spindle

### <h1 align="center">**Gaussian fit function for microtubules' minus end distance to the centriole**</h1>

This script is use to call a pre-written library script that compute for the gaussian fit function on a histogram plot microtubules minus end distance to the center of each spindle pole (centriole). The aim is to determine the radius of the centrosome which corresponds with the second value ($x2$) on the x-axis intercept with y-axis when the value on y-axis is at half the maximum peak height of the gaussian peak.

The gaussian function is computed using the following equation:

# $f(x) = Ae^{\frac{-(x - x0)^2}{2σ^2}}$

**Amplitude (A):**
This is the peak height of the Gaussian curve.
>> The amplitude indicates the maximum value of the density function. In the context of a histogram fitted by a Gaussian curve, it represents the height of the curve at its peak. A higher amplitude means a higher peak, suggesting more data points are concentrated around the mean.

**Mean (x0):**
This is the position of the center of the peak.
>The mean is the average or expected value of the distribution. It indicates where the center of the data distribution is located along the x-axis. In a perfectly symmetric Gaussian distribution, this is also the point of symmetry.

**Standard Deviation (σ):**
This measures the width of the peak.
>The standard deviation is a key measure of the spread or dispersion of the data around the mean. A smaller σ indicates that the data points are clustered closely around the mean, resulting in a narrower peak. Conversely, a larger σ suggests a wider spread of data, leading to a broader peak. It essentially describes how much variation or "dispersion" there is from the average (mean).

### **Instruction on how to run this script**

To run this notebook successfully, this notebook must be in the same folder with the python file called **`MTGaussianFitting.py`** 

The file input and output folder(s) should be define in the second cell of this notebook. Also, the name of the input `.csv` file to be  should be added correctly in the third cell of this notebook. It is important to know that all the imported libraries used in this code should be correctly install in a created python environment. The following lines can be used to install the required libraries to run this notebook:

* **Jupyter lab**
`conda install -c conda-forge jupyterlab`
 
* **Pandas**
`pip install pandas`

* **Numpy**
`conda install -c conda-forge numpy`

* **Matplotlib**
`conda install -c conda-forge matplotlib`

* **Scipy**
`conda install -c conda-forge scipy`

### **Bins size estimation** 
To optimally estimate the bins size of the histogram plot, Freedman Diaconis Estimator function was used in the code. For more detail, see link ([Freedman Diaconis Estimator](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html))
# $h = 2  \frac{IQR}{n^{\frac{1}{3}}}$

* $h$: Freedman Diaconis Estimator
* $n$: number or length of data
* $IQR$: Interquartile Range See link ([Freedman Diaconis Estimator](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html))

The binwidth is proportional to the interquartile range (IQR) and inversely proportional to cube root of data.size. 
Can be too conservative for small datasets, but is quite good for large datasets. The IQR is very robust to outliers.

