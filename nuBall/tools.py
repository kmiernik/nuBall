import numpy
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def progress_bar(n, n_max, time_to_n=None):
    """
    Show progress bar with arrow, completed percentage and optional
    current time and projected time to finish.
    """
    print('\r', end='')
    text = ''
    done = int(n / n_max * 40)
    text += ('[' + '=' * (done - 1) + '>' + '.' * (40 - done - 1) + ']' + 
            ' {:>5.2f}%  '.format((n / n_max) * 100))
    if time_to_n is not None:
        text += '{:>5.1f} s (est. {:>5.1f} s)     '.format(time_to_n, 
                time_to_n * n_max / n)
    print(text, end='', flush=True)


def list_spectra(fin, attrs=False, verbose=True):
    """
    List all spectra and their attributes in a given file or group
    """
    result = []
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            result.append(name)
            if verbose:
                print(name)
            if verbose and attrs:
                for key, val in obj.attrs.items():
                    print('   {} {}'.format(key, val))

    fin.visititems(print_attrs)
    return result


def peaks_function(x, *args):
    """
    x is x-axis of histogram

    args is a list of fit parameters: 
    [a, b, x0, A0, s0, [x1, A1, s1, ...]]

    where a, b are background estimation (ax + b), 
    xi, Ai, si are i-th peak estimations

    The minimal number of arguments is 5
    Number of arguments should fulfill (n - 2) % 3 == 0

    returns parameters and their error
    pars, dpars

    """
    n = len(args)
    if n < 5:
        print('Too few arguments')
        return None
    elif (n - 2) % 3 != 0:
        print('Wrong number of arguments')
        return None
    a = args[0]
    b = args[1]
    y = a * x + b
    for i in range(2, n, 3):
        x0 = args[i]
        A = args[i+1]
        s = args[i+2]
        y += (A / numpy.sqrt(2 * numpy.pi * s**2) 
                * numpy.exp(-(x - x0)**2 / (2 * s)**2))
    return y


def fit(x, c, r, peaks, verbose=True):
    """
    Fit gaussian peaks at listed locations to the histogram counts c
    and x-axis x in range r.

    x - x-axis
    c - y-axis (counts)
    r - range (indexes of x-axis) e.g [230, 290]
    peaks - list of peaks (indexes of x-axis) e.g [248, 253, 267]
    """

    xdata = x[r[0]:r[1]]
    ydata = c[r[0]:r[1]]
    a = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    b = ydata[0] - xdata[0] * a
    args = [a, b]
    for p in peaks:
        args.append(p)
        args.append(c[int(p)])
        args.append(1.0)
    popt, pconv = curve_fit(peaks_function, xdata, ydata, p0=[*args])
    pars = []
    dp = []
    n = len(popt)
    for i in range(n):
        pars.append(popt[i])
        dp.append(numpy.sqrt(pconv[i, i]))
        if not verbose:
            continue
        if i == 0:
            print('a', end=' ')
        elif i == 1:
            print('b', end=' ')
        else:
            j = int((i - 2) // 3)
            k = int((i - 2) % 3)
            if k % 3 == 0:
                print('x{}'.format(j), end=' ')
            elif k % 3 == 1:
                print('A{}'.format(j), end=' ')
            else:
                print('s{}'.format(j), end=' ')
        print('\t{:.3f}\t {:.3f}'.format(pars[-1], dp[-1]))
    return pars, dp


def plot(ax, dataset):
    """
    Convenient wrapper for matplotlib plot function, puts
    histogram in an given ax (plot is labeled with dataset name)
    """
    y = numpy.array(dataset)
    x = numpy.arange(dataset.shape[0]) + 0.5
    ax.plot(x, y, ds='steps-mid', label='{}'.format(dataset.name))


if __name__ == '__main__':
    pass
