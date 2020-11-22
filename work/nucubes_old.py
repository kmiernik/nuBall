"""
Tools to make gates on ge-ge-ge, ge-ge-la and ge-la-la coincidences
Input file must be organized as follow:
    HDF5 dataset GGG_list
    matrix n * 7, STD_U32LE

    E0 E1 E2 T0 T1 T2 pattern
    
    where pattern is mulitplicity * 100 + hit_pattern

    This version is for multiplicity and open coin window
"""
import argparse
import h5py
import numpy
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
 

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



def gegege(dataset, gate_z, gate_y=None, gate_t=None, gate_m=None,
           D=(4096, 4096), units=(100, 100), chunk_size=10000000):
    """
    Calculate Ge-Ge-Ge 2D or 1D matrix based on selected gate on Z axis 
    (2D array is returned), and/or Y-axis (1D), and/or time gate (1D). 
    D is dimension of array (1 keV per channel) and chunk_size is the 
    chunk of input file loaded at a time. It is best to keep it as large
    as possible, but reduce it if you run into memory issues.
    """

    n = dataset.shape[0]
    matrix = numpy.zeros((D[0], D[1]))
    left_pos = 0
    is_something_left = True
    t0 = datetime.datetime.now()
    progress_bar(left_pos, n)
    while is_something_left:
        right_pos = left_pos + chunk_size
        if right_pos > n:
            right_pos = n
            is_something_left = False
        submatrix = dataset[left_pos:right_pos, :]

        pattern = numpy.equal(submatrix[:, 6] % 100, 
                            numpy.ones(len(submatrix)) * 7)
        for loc in [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], 
                [2, 0, 1], [2, 1, 0]]:
            logic = numpy.logical_and(pattern,
                            numpy.logical_and(
                                submatrix[:, loc[0]] >= gate_z[0] * units[0], 
                                submatrix[:, loc[0]] < gate_z[1] * units[1])
                        )
            if gate_m is not None:
                logic = numpy.logical_and(logic,
                            numpy.logical_and(
                                    submatrix[:, 6] // 100 >= gate_m[0],
                                    submatrix[:, 6] // 100 <= gate_m[1])
                            )
            if gate_y is not None:
                logic = numpy.logical_and(logic,
                            numpy.logical_and(
                                submatrix[:, loc[1]] >= gate_y[0] * units[0], 
                                submatrix[:, loc[1]] < gate_y[1] * units[1]))

            if gate_t is not None:
                logic = numpy.logical_and(logic, 
                            numpy.logical_and(
                                submatrix[:, loc[0]+3] >= gate_t[0] * 1000, 
                                submatrix[:, loc[0]+3] < gate_t[1] * 1000)
                            )
                logic = numpy.logical_and(logic, 
                            numpy.logical_and(
                                submatrix[:, loc[1]+3] >= gate_t[0] * 1000, 
                                submatrix[:, loc[1]+3] < gate_t[1] * 1000)
                            )
                logic = numpy.logical_and(logic, 
                            numpy.logical_and(
                                submatrix[:, loc[2]+3] >= gate_t[0] * 1000, 
                                submatrix[:, loc[2]+3] < gate_t[1] * 1000)
                            )
            logic = numpy.logical_not(logic)
            mask = numpy.repeat(logic, submatrix.shape[1])
            masked = numpy.ma.array(submatrix, mask=mask)

            gg, xe, ye = numpy.histogram2d(
                                 masked[:, loc[1]].compressed() / units[0], 
                                 masked[:, loc[2]].compressed() / units[1], 
                                        bins=[D[0], D[1]], 
                                        range=[[0, D[0]], [0, D[1]]])
            matrix += gg

        left_pos = right_pos
        dt = (datetime.datetime.now() - t0).total_seconds()
        progress_bar(left_pos, n, dt)

    #mT = numpy.copy(matrix.transpose())
    #numpy.fill_diagonal(mT, 0)
    #matrix += mT

    print()
    print('Events in matrix:', matrix.sum().sum())
    if gate_y is None:
        return matrix
    else:
        return matrix.sum(axis=0)


def lagege(dataset, gate_t, gate_z=None, gate_y=None, D=(4096, 4096),
        units=(100, 100), chunk_size=10000000):
    """
    Calculate Ge-Ge-LaBr 2D or 1D matrix based on selected gate on time
    (on LaBr), gate_z is set on LaBr, gate_y on Ge. 2D Ge-Ge or 1D Ge spectrum
    is returned
    """

    n = dataset.shape[0]
    matrix = numpy.zeros((D[0], D[1]))
    left_pos = 0
    is_something_left = True
    t0 = datetime.datetime.now()
    progress_bar(left_pos, n)
    while is_something_left:
        right_pos = left_pos + chunk_size
        if right_pos > n:
            right_pos = n
            is_something_left = False
        submatrix = dataset[left_pos:right_pos, :]

        # Pattern number La, Ge, Ge
        patterns = [[6, [0, 1, 2]], [5, [1, 0, 2]], [3, [2, 0, 1]]]
        for pattern in patterns:
            hit_pattern = pattern[0]
            loc = pattern[1]
            logic = numpy.logical_and(
                        numpy.equal(submatrix[:, 6], 
                                numpy.ones(len(submatrix)) * hit_pattern),
                        numpy.logical_and(
                            submatrix[:, loc[0]+3] >= gate_t[0] * 1000, 
                            submatrix[:, loc[0]+3] < gate_t[1] * 1000)
                        )
            if gate_z is not None:
                logic = numpy.logical_and(
                                logic,
                                numpy.logical_and(
                                submatrix[:, loc[0]] >= gate_z[0] * units[0], 
                                submatrix[:, loc[0]] < gate_z[1] * units[1])
                            )
            if gate_y is not None:
                logic = numpy.logical_and(
                                logic,
                                numpy.logical_and(
                                submatrix[:, loc[1]] >= gate_y[0] * units[0], 
                                submatrix[:, loc[1]] < gate_y[1] * units[1])
                            )
            logic = numpy.logical_not(logic)
            mask = numpy.repeat(logic, submatrix.shape[1])
            masked = numpy.ma.array(submatrix, mask=mask)

            gg, xe, ye = numpy.histogram2d(
                                masked[:, loc[1]].compressed() / units[0], 
                                masked[:, loc[2]].compressed() / units[1], 
                                        bins=[D[0], D[1]], 
                                        range=[[0, D[0]], [0, D[1]]])
            matrix += gg

        left_pos = right_pos
        dt = (datetime.datetime.now() - t0).total_seconds()
        progress_bar(left_pos, n, dt)

    print()
    if gate_z is None and gate_y is None:
        print('Events in matrix:', matrix.sum().sum())
        return matrix
    else:
        matrix = matrix.sum(axis=0)
        print('Events in matrix:', matrix.sum())
        return matrix



def gelala(dataset, gate_z, gate_y, gate_x=None, D=(4096, 800), 
        units=(1000, 100), chunk_size=10000000):
    """
    Calculate Ge-LaBr-LaBr 1D matrix based on selected gate on Z axis (Ge)
    and two LaBr (y, x), returns time difference between two LaBr.
    """

    n = dataset.shape[0]
    matrix = numpy.zeros((D[0], D[1]))
    left_pos = 0
    is_something_left = True
    t0 = datetime.datetime.now()
    progress_bar(left_pos, n)
    while is_something_left:
        right_pos = left_pos + chunk_size
        if right_pos > n:
            right_pos = n
            is_something_left = False
        submatrix = dataset[left_pos:right_pos, :]

        # Pattern number Ge, La, La
        patterns = [[1, [0, 1, 2]], [1, [0, 2, 1]], 
                    [2, [1, 0, 2]], [2, [1, 2, 0]],
                    [4, [2, 0, 1]], [4, [2, 1, 0]]]
        for pattern in patterns:
            hit_pattern = pattern[0]
            loc = pattern[1]
            logic = numpy.logical_and(
                            numpy.equal(submatrix[:, 6], 
                                numpy.ones(len(submatrix)) * hit_pattern),
                            numpy.logical_and(
                                submatrix[:, loc[0]] >= gate_z[0] * units[1], 
                                submatrix[:, loc[0]] < gate_z[1] * units[1])
                            )
            logic = numpy.logical_and(
                            logic,
                            numpy.logical_and(
                                submatrix[:, loc[1]] >= gate_y[0] * units[1], 
                                submatrix[:, loc[1]] < gate_y[1] * units[1])
                            )
            if gate_x is not None:
                logic = numpy.logical_and(
                                logic,
                                numpy.logical_and(
                                submatrix[:, loc[2]] >= gate_x[0] * units[1], 
                                submatrix[:, loc[2]] < gate_x[1] * units[1])
                                )
            logic = numpy.logical_not(logic)
            mask = numpy.repeat(logic, submatrix.shape[1])
            masked = numpy.ma.array(submatrix, mask=mask)

            gg, xe, ye = numpy.histogram2d(
                                masked[:, loc[2]].compressed() / units[1], 
                                (masked[:, loc[2]+3].compressed() - 
                                masked[:, loc[1]+3].compressed()) / units[0],
                                        bins=[D[0], D[1]], 
                                        range=[[0, D[0]], [0, 100]])
            matrix += gg

        left_pos = right_pos
        dt = (datetime.datetime.now() - t0).total_seconds()
        progress_bar(left_pos, n, dt)

    print()
    print('Events in matrix:', matrix.sum().sum())
    if gate_x is None:
        return matrix
    else:
        matrix = matrix.sum(axis=0)
        print('Events in matrix:', matrix.sum())
        return matrix


def rebin2D(matrix, bin_size):
    """ 
    Rebins matrix by factor bin_size in each dimension. Be careful:
    matrix size must divide by bin_size
    Returns rebinned matrix and axes for convinence
    """
    Dx = matrix.shape[0]
    Dy = matrix.shape[1]
    matrix = numpy.reshape(matrix, (int(Dx / bin_size[0]), 
        bin_size[0], int(Dy / bin_size[1]), bin_size[1])).mean(3).mean(1)
    x_ax = numpy.arange(int(Dx / bin_size[0])) * bin_size[0]
    y_ax = numpy.arange(int(Dy / bin_size[1])) * bin_size[1]
    return matrix, x_ax, y_ax


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NuBall 3D matrix gating script, default gates are set on Ge-Ge-Ge events')
    parser.add_argument('-z', nargs=2, 
                        dest='gate_z', type=float, help='Z gate')
    parser.add_argument('-y', nargs=2, 
                        dest='gate_y', type=float, help='Y gate')
    parser.add_argument('-x', nargs=2, 
                        dest='gate_x', type=float, help='Y gate')
    parser.add_argument('-t', nargs=2, 
                        dest='gate_t', type=float, help='Time gate')
    parser.add_argument('-m', nargs=2, 
                        dest='gate_m', type=float, help='Multiplicity gate')
    parser.add_argument('--logz', action='store_true', 
                        help='Use logarithmic scale for number of counts in a 2D matrix')
    parser.add_argument('--no_plot', action='store_true', 
                        help='Disable plotting')
    parser.add_argument('--save', nargs=1, 
                        help='Name of output text file')

    parser.add_argument('--chunk', type=int, default=10000000, 
            help='Size of data chunk loaded at time (default = 1e7), the larger the better, but reduce it if you run into memory issues')
    parser.add_argument('-D', type=int, nargs=2, default=[4096, 4096],
            help='Matrix dimension')
    parser.add_argument('-r', type=int, nargs=2, default=[4, 4], 
            help='Rebinning factor for 2D matrices (default = 4x4)')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--lagege', action='store_true', 
            help='Use LaBr-Ge-Ge events, -t must be set')
    mode.add_argument('--gelala', action='store_true', 
            help='Use Ge-LaBr-LaBr events -z, and -y must be set')

    parser.add_argument('input', help='Input HDF5 file')
    args = parser.parse_args()


    fin = h5py.File(args.input, 'r')

    dataset = fin['GGG_list']
    print('# Number of entries', dataset.shape[0])

    if args.lagege:
        if args.gate_t is None:
            print('When using LaBr-Ge-Ge gate, you must use -t option')
            exit()
        matrix = lagege(dataset, args.gate_t, args.gate_z, args.gate_y, args.D,
                chunk_size=args.chunk)
    elif args.gelala:
        if args.gate_z is None or args.gate_y is None:
            print('When using Ge-LaBr-LaBr gate, you must use -z, and -y options')
            exit()
        matrix = gelala(dataset, args.gate_z, args.gate_y, args.gate_x, 
                        chunk_size=args.chunk)
    else:
        if args.gate_z is None:
            print('When using Ge-Ge-Ge gate, you must use -z option')
            exit()
        matrix = gegege(dataset, args.gate_z, args.gate_y, args.gate_t, 
                args.gate_m, args.D, chunk_size=args.chunk)

    fin.close()


    if args.save is not None:
        header = 'File {},'.format(args.input)
        if args.gelala:
            header += ' Ge-LaBr-LaBr,'
        elif args.lagege:
            header += ' LaBr-Ge-Ge,'
        else:
            header += ' Ge-Ge-Ge,'
        if args.gate_z is not None:
            header += ' gate z=[{0[0]}, {0[1]}],'.format(args.gate_z)
        if args.gate_y is not None:
            header += ' gate y=[{0[0]}, {0[1]}],'.format(args.gate_y)
        if args.gate_x is not None:
            header += ' gate x=[{0[0]}, {0[1]}],'.format(args.gate_x)
        if args.gate_t is not None:
            header += ' gate t=[{0[0]}, {0[1]}],'.format(args.gate_t)
        if args.gate_m is not None:
            header += ' gate m=[{0[0]}, {0[1]}],'.format(args.gate_m)
        numpy.savetxt(args.save[0], matrix, fmt="%d", header=header)

    if args.no_plot:
        pass
    else:
        if matrix.ndim == 2:
            plt.figure(1, (10, 8))
            matrix, x_ax, y_ax = rebin2D(matrix, args.r)
            if args.logz:
                matrix = numpy.ma.masked_where(matrix <= 0, numpy.log10(matrix))
            plt.pcolormesh(x_ax, y_ax, matrix, cmap=cm.RdYlGn_r)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.colorbar()
        else:
            plt.figure(1, (10, 6))
            z_ax = numpy.arange(matrix.shape[0])
            plt.plot(z_ax, matrix, ds='steps-mid')
            plt.ylim(0, None)
            plt.xlim(0, z_ax[-1])
            plt.xlabel('Z')

        plt.tight_layout()
        plt.show()

