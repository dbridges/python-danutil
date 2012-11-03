from __future__ import print_function

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import webbrowser as wb

from itertools import groupby
from inspect import getargspec
from scipy.stats import fprob
from scipy import stats
from scipy.ndimage import imread
from dateutil.parser import parse as parse_date 

### General I/O

def import_file(filename, datatype='', *args, **kwargs):
    """
    filename : string
        File to import.

    datatype : string
        Use 'data' to import file as a data file.

    Imports text and images files, automatically seperating text into arrays
    based on the files extension, or optional type paramater.
    """
    if filename.endswith(('.png', '.bmp', '.jpeg', '.jpg', '.eps', '.tiff')):
        return imread(filename, *args, **kwargs)
    if filename.endswith('.csv') or datatype == 'data':
        return _import_data(filename, *args, **kwargs)
    with open(filename) as f:
        data = f.read()
        f.close()
        return data

def _import_data(filename, delimiter=None):
    """
    filename : string
        Filename of data file to import.

    delimiter : string
        Text sequence which separates data entries.
        This is inferred to be a tab for .txt files
        and a comma for .csv files.
    """
    if filename.endswith('.txt') and delimiter == None:
        delimiter = '\t'
    elif filename.endswith('.csv') and delimiter == None:
        delimiter = ','
    if delimiter == None:
        raise ValueError('Delimiter could not be inferred and' \
                         ' was not specified.')
    f = open(filename, 'r')
    data_out = []
    for line in f:
        new_row = line.split(delimiter)
        new_row = [item.strip('\n') for item in new_row]
        # Try to convert items to floats, if not leave as strings
        for n in range(len(new_row)):
            try:
                if '.' in new_row[n] or 'e' in new_row[n] or (
                        'E' in new_row[n]):
                    new_row[n] = float(new_row[n])
                else:
                    new_row[n] = int(new_row[n])
            except:
                pass
        data_out.append(new_row)
    # get rid of the trailing newlines
    for n in range(len(data_out)-1, 0, -1):
        if data_out[n] != ['']:
            break

    return data_out[:n+1]

def print_table(input_list, headings=None, usetabs=False, fmt='%g'):
    """
    input_list : array like
        2D list to print out in table form.

    headings : list
        1D list of table headings.

    usetabs : bool
        Whether to seperate entries with tabs or spaces. Useful
        if you want to copy the data into another program.

    fmt : string
        Format string to use when printing numbers.
    """
    if not getattr(input_list[0], '__iter__', False):
        if headings != None:
            for item in headings:
                print(item, end='\t')
            print('\n', end='')
        for item in input_list:
            if isnumber(item):
                print(fmt % item)
            else:
                print(item)
        return
   
    if usetabs:
        if headings != None:
            for item in headings:
                print(item, end='\t')
            print('\n', end='')
        if not getattr(input_list[0], '__iter__', False):
            for item in input_list:
                if isnumber(item):
                    print(fmt % item)
                else:
                    print(item)
            return
        else:
            for row in input_list:
                for item in row:
                    if isnumber(item):
                        print(fmt % item, end='\t')
                    else:
                        print(item, end='\t')
                print('\n', end='')
            return
    else:
        # convert to list and find longest item in each column
        # can only handle 50 columns
        col_width = [0]*50 
        for i in range(len(input_list)):
            for j in range(len(input_list[i])):
                if isnumber(input_list[i][j]):
                    input_list[i][j] = fmt % input_list[i][j]
                else:
                    input_list[i][j] = str(input_list[i][j])
                if len(input_list[i][j]) > col_width[j]:
                    col_width[j] = len(input_list[i][j])
        
        if headings != None:
            for j in range(len(headings)):
                if len(headings[j]) > col_width[j]:
                    col_width[j] = len(headings[j])
                print(headings[j], 
                      end=' '*(col_width[j]-len(headings[j])+3))
            print('\n', end='')
        
        for i in range(len(input_list)):
            for j in range(len(input_list[i])):
                print(input_list[i][j], 
                      end=' '*(col_width[j]-len(input_list[i][j])+3))
            print('\n', end='')

def savenpy(filename, X, delimiter=','):
    """
    filename : string
        Location to save file.

    X : Numpy Array
        Array to be written to file

    Saves a numpy array to file.
    """
    np.savetxt(filename, X, delimiter=delimiter, fmt='%8f')


def savecsv(fname, seq, headers=None):
    """
    fname : string
        Location to save file.

    seq : 2D array of objects
        List of rows to be written.

    headers : list of strings
        Headers to be written at top of file.

    Writes a csv file with the string representation of the data in seq.
    Data does not have to be of the same data type.
    """
    if not getattr(seq[0], '__iter__', False):
        # 1d array
        f = open(fname, 'w')
        if headers != None:
            line = ''
            for item in headers:
                line += str(item)
                line += ','
            line = line[:-1] # remove trailing comma
            line += '\n'
            f.write(line)
        for i in range(len(seq)):
            f.write(str(seq[i])+'\n')
        f.close()
    else:
        f = open(fname, 'w')
        if headers != None:
            line = ''
            for item in headers:
                line += str(item)
                line += ','
            line = line[:-1]
            line += '\n'
            f.write(line)
        for row in seq:
            line = ''
            for item in row:
                line += str(item) + ','
            line = line[:-1]
            f.write(line + '\n')
        f.close()

#
### List Operations
#

def all_indices(l, i):
    """
    l : list
        list of items
    i : object
        item to look for in list

    Returns a list of all indicies for item i.
    """ 
    indicies = []
    for n in range(len(l)):
        if l[n] == i:
            indicies.append(n)
    return indicies

def delete_duplicates(seq):
    """
    seq : list
        The sequence to delete duplicates from.

    Returns a list with duplicates deleted from seq while maintaining ordering.
    """
    output = []
    for item in seq:
        if item not in output:
            output.append(item)
    return output

def split(seq, key=None):
    """
    seq : list
        The sequence to split.

    key : callable
        Split using value returned by applying key to item.

    Splits list into runs of identical elements as specified by key.

    Example:
    This splits data based on first item in each list.

    data = [['a', 1, 2], ['a', 3, 4], ['b', 1, 2]]
    split(data, key=itemgetter(0))
    
    outputs: [[['a', 1, 2], ['a', 3, 4]], [['b', 1, 2]]]
    """
    output = []
    [output.append(list(run)) for item, run in groupby(seq, key)]
    return output

def pack(*args):
    """
    *args : list
        A series of lists.

    pack(x, y) combines lists [x1, x2, x3], [y1, y2, y3] 
    into form [[x1, y1], [x2, y2], [x3, y3]]
    It is essentially just the builtin zip.
    """
    return [list(a) for a in zip(*args)] 

def unpack(seq, column):
    """
    seq : list
        A packed list of form [[x1, y1], [x2, y2], [x3, y3]]

    Extract column in packed list.
    With input seq = [[x1, y1], [x2, y2], [x3, y3]]
    unpack(seq, 0) returns [x1, x2, x3]
    """
    return list(zip(*seq)[column])

def natural_sort(seq): 
    """
    seq : list
        A list of strings to sort.

    Sorts the given sequence in place.
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    seq.sort(key=alphanum_key) 

### Utilities

def docs():
    """
    Opens a documentation browsers for numpy, scipy, matplotlib, and danutil.
    """
    wb.open('http://github.com/dbridges/python-danutil')

def isnumber(val):
    """
    val : object
        The object to be tested

    Returns True of val is an int, long, float, or complex number.
    """
    return isinstance(val, (int, long, float, complex))

def listcwd(startswith='', endswith=''):
    """
    starts : string
        Filter results such that they start with this string.
        This can be a tuple of strings.
    
    ends : string
        Filter resutls such that they end with this string.
        This can be a tuple of strings.
    
    Returns a list of files in the current working directory that
    start with 'starts' and end with 'ends'
    """
    files = os.listdir(os.getcwd())
    files = [f for f in files if f.startswith(startswith)
                                 and f.endswith(endswith)]
    return files

#
### Plotting Convenience 
#

def date_plot(dates, data, *args, **kwargs):
    """
    dates : list
        A 1D list of date strings.

    data : list
        A 1D list of data corresponding to the values in dates.
    
    Plots dates vs. data.
    """
    fig, ax = plt.subplots(1)
    dates = [parse_date(date) for date in dates]
    ax.plot(dates, data, *args, **kwargs)
    fig.autofmt_xdate()

def plot_function(func, coef, xmin, xmax, num_vals=100, *args):
    """
    func : callable
        The function to be plotted.

    coef : list
        List of coefficients for function.

    xmin : float
        The minimum x value to start plotting at.

    xmax : float
        The maximum x value to plot to.

    num_vals : int
        The number of discrete values to plot to.

    Plots the function to the current matplotlib plot.

    Example:
    
    def _line_func(x, m, b):
        return m * x + b

    x = np.linspace(0, 10, 10)
    y = [0.1, 1.3, 2.1, 3.2, 4.7, 5.6, 6.4, 7.5,9, 10.1]

    fit = scipy.optimize.curve_fit(line_func, x, y)
    plot_function(line_func, fit[0], 0, 10)
    """
    a = getargspec(func)
    if (len(coef) + 1 != len(a.args)):
        raise ValueError('The number of coefficients does not match the ' \
                         'number of arguments to func')
    xx = np.linspace(xmin, xmax, num_vals)
    plt.plot(xx, [func(x, *coef) for x in xx], *args)

#
### Curve Fitting
#

def fitline(x, y):
    """
    Fits the data x, y to a line.

    Returns (slope, x_intercept, r_squared)
    """
    fit = stats.linregress(x, y)
    return (fit[0], fit[1], fit[2]**2)

#
### Mathematical Functions for Curve Fitting 
#

def exp_func(x, a, b, c):
    """
    Returns a function of form: a*e^(b*x) + c
    """
    return a * np.exp(b * x) + c

def gaussian_func(x, a, b, c):
    """
    aReturns a function of form: *e^((x-b)^2/2c^2)

    a = 1/(sigma*sqrt(2*pi))    
    b = mu                      (expected value)
    c = sigma                   (standard deviation)
    """
    return a * np.exp(-((x-b)**2)/(2*c**2))

def line_func(x, m, b):
    """
    Returns a function of form: m*x + b
    """
    return m * x + b

def sin_func(x, a, w, phi):
    """
    aReturns a function of form: *sin(w*x + phi)
    """
    return a * np.sin(w*x + phi) 

#
### Statistics
#

def std(seq, *args, **kwargs):
    """
    seq : array
        The sequence to compute the standard deviation of.

    Returns the sample standard deviation of seq.
    """
    return np.std(seq, ddof=1, *args, **kwargs)

class ANOVA(object):
    """
    Calculates the one-way ANOVA of passed samples.

    ANOVA(list1, list2, ...)

    Attributes:
        self.grand_mean         Grand mean of all samples
        self.sst                Total sum of squares
        self.ssb                Between groups sum of squares
        self.ssw                Within group sum of squares
        self.fstat              F statistic
        self.pvalue             P value

    Methods:
        self.summary()          Prints the ANOVA summary table.
    """
    def __init__(self, *args):
        super(ANOVA, self).__init__()
        samples = [np.asarray(x) for x in args]
        all_samples = np.concatenate(samples)
        self.grand_mean = np.mean(all_samples)
        self.sst = np.sum([(x - self.grand_mean)**2 for x in all_samples])
        self.ssb = np.sum(
                    [len(x)*(np.mean(x) - self.grand_mean)**2 for x in samples]
                    ) 
        self.ssw = self.sst - self.ssb
        self.N = len(all_samples)
        self.k = len(samples) 
        self.ssbdf = self.k - 1
        self.sswdf = self.N - self.k
        self.mssb = self.ssb / self.ssbdf
        self.mssw = self.ssw / self.sswdf
        self.fstat = self.mssb / self.mssw
        self.pvalue = fprob(self.ssbdf, self.sswdf, self.fstat) 

    def summary(self):
        """docstring for print_table"""
        print_table([['Between', self.ssb, self.ssbdf, 
                        self.mssb, self.fstat, self.pvalue],
                     ['Within', self.ssw, self.sswdf, 
                        self.mssw],
                     ['Total', self.sst, self.N - 1]],
                     headings=['', 'SS', 'df', 'MS', 'F stat', 'p value'])

