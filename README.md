# python-danutil
* [ General I/O](# General I/O)
* [ List Operations](# List Operations)
* [  Utilities](#  Utilities)
* [ Plotting Convenience](# Plotting Convenience)
* [ Curve Fitting](# Curve Fitting)
* [ Mathematical Functions for Curve Fitting](# Mathematical Functions for Curve Fitting)
* [ Statistics](# Statistics)
<a id=" General I/O">## General I/O</a>

```
import_file(filename, datatype='', *args, **kwargs)

    filename : string
        File to import.

    datatype : string
        Use 'data' to import file as a data file.

    Imports text and images files, automatically seperating text into arrays
    based on the files extension, or optional type paramater.
    
```

```
print_table(input_list, headings=None, usetabs=False, fmt='%g')

    input_list : array like
        2D list to print out in table form.

    headings : list
        1D list of table headings.

    usetabs : bool
        Whether to seperate entries with tabs or spaces. Useful
        if you want to copy the data into another program.

    fmt : string
        Format string to use when printing numbers.
    
```

```
savenpy(filename, X, delimiter=',')

    filename : string
        Location to save file.

    X : Numpy Array
        Array to be written to file

    Saves a numpy array to file.
    
```

```
savecsv(fname, seq, headers=None)

    fname : string
        Location to save file.

    seq : 2D array of objects
        List of rows to be written.

    headers : list of strings
        Headers to be written at top of file.

    Writes a csv file with the string representation of the data in seq.
    Data does not have to be of the same data type.
    
```

<a id=" List Operations">## List Operations</a>

```
all_indices(l, i)

    l : list
        list of items
    i : object
        item to look for in list

    Returns a list of all indicies for item i.
    
```

```
delete_duplicates(seq)

    seq : list
        The sequence to delete duplicates from.

    Returns a list with duplicates deleted from seq while maintaining ordering.
    
```

```
split(seq, key=None)

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
    
```

```
pack(*args)

    *args : list
        A series of lists.

    pack(x, y) combines lists [x1, x2, x3], [y1, y2, y3] 
    into form [[x1, y1], [x2, y2], [x3, y3]]
    It is essentially just the builtin zip.
    
```

```
unpack(seq, column)

    seq : list
        A packed list of form [[x1, y1], [x2, y2], [x3, y3]]

    Extract column in packed list.
    With input seq = [[x1, y1], [x2, y2], [x3, y3]]
    unpack(seq, 0) returns [x1, x2, x3]
    
```

```
natural_sort(seq)

    seq : list
        A list of strings to sort.

    Sorts the given sequence in place.
    
```

<a id="  Utilities">##  Utilities</a>

```
docs()

    Opens a documentation browsers for numpy, scipy, matplotlib, and danutil.
    
```

```
isnumber(val)

    val : object
        The object to be tested

    Returns True of val is an int, long, float, or complex number.
    
```

```
listcwd(startswith='', endswith='')

    starts : string
        Filter results such that they start with this string.
        This can be a tuple of strings.
    
    ends : string
        Filter resutls such that they end with this string.
        This can be a tuple of strings.
    
    Returns a list of files in the current working directory that
    start with 'starts' and end with 'ends'
    
```

<a id=" Plotting Convenience">## Plotting Convenience</a>

```
date_plot(dates, data, *args, **kwargs)

    dates : list
        A 1D list of date strings.

    data : list
        A 1D list of data corresponding to the values in dates.
    
    Plots dates vs. data.
    
```

```
plot_function(func, coef, xmin, xmax, num_vals=100, *args)

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
    
```

<a id=" Curve Fitting">## Curve Fitting</a>

```
fitline(x, y)

    Fits the data x, y to a line.

    Returns (slope, x_intercept, r_squared)
    
```

<a id=" Mathematical Functions for Curve Fitting">## Mathematical Functions for Curve Fitting</a>

```
exp_func(x, a, b, c)

    Returns a function of form: a*e^(b*x) + c
    
```

```
gaussian_func(x, a, b, c)

    aReturns a function of form: *e^((x-b)^2/2c^2)

    a = 1/(sigma*sqrt(2*pi))    
    b = mu                      (expected value)
    c = sigma                   (standard deviation)
    
```

```
line_func(x, m, b)

    Returns a function of form: m*x + b
    
```

```
sin_func(x, a, w, phi)

    aReturns a function of form: *sin(w*x + phi)
    
```

<a id=" Statistics">## Statistics</a>

```
std(seq, *args, **kwargs)

    seq : array
        The sequence to compute the standard deviation of.

    Returns the sample standard deviation of seq.
    
```

```
ANOVA(object)

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
    
```

