"""
Tools to make gates on ge-ge coincidences
Input file must be organized as follow:
    HDF5 dataset GG_list
    matrix n * 6, STD_U16LE

    E0 E1 T0 T1 L0 L1

"""
import argparse
import h5py
import numpy
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm

# Ge detectors, ID : theta, phi, r
locations = {
    2 : [2.3314457213990654, 0.3143337982841788, 160.816],
    4 : [2.3316726142018247, 0.9426523290021372, 160.816],
    6 : [2.331794787249464, 1.5707963267948966, 160.816],
    8 : [2.331794787249464, 2.1992893904380546, 160.816],
    10 : [2.3316726142018247, 2.827782454081213, 160.816],
    12 : [2.3314457213990654, 3.4554028530983736, 160.816],
    14 : [2.3312013753037863, 4.083895916741532, 160.816],
    16 : [2.3310617489636263, 4.71238898038469, 160.816],
    18 : [2.3310617489636263, 5.340532978177449, 160.816],
    20 : [2.3312013753037863, 5.9688515088954075, 160.816],
    25 : [1.8238690683340746, 0.2617993877991494, 173.024],
    26 : [1.8238690683340746, 0.2617993877991494, 173.024],
    27 : [1.8238690683340746, 0.2617993877991494, 173.024],
    28 : [1.8238690683340746, 0.2617993877991494, 173.024],
    31 : [1.8238690683340746, 0.7853981633974483, 173.024],
    32 : [1.8238690683340746, 0.7853981633974483, 173.024],
    33 : [1.8238690683340746, 0.7853981633974483, 173.024],
    34 : [1.8238690683340746, 0.7853981633974483, 173.024],
    37 : [1.8238690683340746, 1.3089969389957472, 173.024],
    38 : [1.8238690683340746, 1.3089969389957472, 173.024],
    39 : [1.8238690683340746, 1.3089969389957472, 173.024],
    40 : [1.8238690683340746, 1.3089969389957472, 173.024],
    43 : [1.8238690683340746, 1.8325957145940461, 173.024],
    44 : [1.8238690683340746, 1.8325957145940461, 173.024],
    45 : [1.8238690683340746, 1.8325957145940461, 173.024],
    46 : [1.8238690683340746, 1.8325957145940461, 173.024],
    49 : [1.8238690683340746, 2.356194490192345, 173.024],
    50 : [1.8238690683340746, 2.356194490192345, 173.024],
    51 : [1.8238690683340746, 2.356194490192345, 173.024],
    52 : [1.8238690683340746, 2.356194490192345, 173.024],
    55 : [1.8238690683340746, 2.8797932657906435, 173.024],
    56 : [1.8238690683340746, 2.8797932657906435, 173.024],
    57 : [1.8238690683340746, 2.8797932657906435, 173.024],
    58 : [1.8238690683340746, 2.8797932657906435, 173.024],
    61 : [1.8238690683340746, 3.4033920413889422, 173.024],
    62 : [1.8238690683340746, 3.4033920413889422, 173.024],
    63 : [1.8238690683340746, 3.4033920413889422, 173.024],
    64 : [1.8238690683340746, 3.4033920413889422, 173.024],
    67 : [1.8238690683340746, 3.9269908169872414, 173.024],
    68 : [1.8238690683340746, 3.9269908169872414, 173.024],
    69 : [1.8238690683340746, 3.9269908169872414, 173.024],
    70 : [1.8238690683340746, 3.9269908169872414, 173.024],
    73 : [1.8238690683340746, 4.4505895925855405, 173.024],
    74 : [1.8238690683340746, 4.4505895925855405, 173.024],
    75 : [1.8238690683340746, 4.4505895925855405, 173.024],
    76 : [1.8238690683340746, 4.4505895925855405, 173.024],
    79 : [1.8238690683340746, 4.974188368183839, 173.024],
    80 : [1.8238690683340746, 4.974188368183839, 173.024],
    81 : [1.8238690683340746, 4.974188368183839, 173.024],
    82 : [1.8238690683340746, 4.974188368183839, 173.024],
    85 : [1.8238690683340746, 5.497787143782138, 173.024],
    86 : [1.8238690683340746, 5.497787143782138, 173.024],
    87 : [1.8238690683340746, 5.497787143782138, 173.024],
    88 : [1.8238690683340746, 5.497787143782138, 173.024],
    91 : [1.8238690683340746, 6.021385919380437, 173.024],
    92 : [1.8238690683340746, 6.021385919380437, 173.024],
    93 : [1.8238690683340746, 6.021385919380437, 173.024],
    94 : [1.8238690683340746, 6.021385919380437, 173.024],
    97 : [1.3177235852557188, 0.2617993877991494, 173.024],
    98 : [1.3177235852557188, 0.2617993877991494, 173.024],
    99 : [1.3177235852557188, 0.2617993877991494, 173.024],
    100 : [1.3177235852557188, 0.2617993877991494, 173.024],
    103 : [1.3177235852557188, 0.7853981633974483, 173.024],
    104 : [1.3177235852557188, 0.7853981633974483, 173.024],
    105 : [1.3177235852557188, 0.7853981633974483, 173.024],
    106 : [1.3177235852557188, 0.7853981633974483, 173.024],
    109 : [1.3177235852557188, 1.3089969389957472, 173.024],
    110 : [1.3177235852557188, 1.3089969389957472, 173.024],
    111 : [1.3177235852557188, 1.3089969389957472, 173.024],
    112 : [1.3177235852557188, 1.3089969389957472, 173.024],
    115 : [1.3177235852557188, 1.8325957145940461, 173.024],
    116 : [1.3177235852557188, 1.8325957145940461, 173.024],
    117 : [1.3177235852557188, 1.8325957145940461, 173.024],
    118 : [1.3177235852557188, 1.8325957145940461, 173.024],
    121 : [1.3177235852557188, 2.356194490192345, 173.024],
    122 : [1.3177235852557188, 2.356194490192345, 173.024],
    123 : [1.3177235852557188, 2.356194490192345, 173.024],
    124 : [1.3177235852557188, 2.356194490192345, 173.024],
    127 : [1.3177235852557188, 2.8797932657906435, 173.024],
    128 : [1.3177235852557188, 2.8797932657906435, 173.024],
    129 : [1.3177235852557188, 2.8797932657906435, 173.024],
    130 : [1.3177235852557188, 2.8797932657906435, 173.024],
    133 : [1.3177235852557188, 3.4033920413889422, 173.024],
    134 : [1.3177235852557188, 3.4033920413889422, 173.024],
    135 : [1.3177235852557188, 3.4033920413889422, 173.024],
    136 : [1.3177235852557188, 3.4033920413889422, 173.024],
    139 : [1.3177235852557188, 3.9269908169872414, 173.024],
    140 : [1.3177235852557188, 3.9269908169872414, 173.024],
    141 : [1.3177235852557188, 3.9269908169872414, 173.024],
    142 : [1.3177235852557188, 3.9269908169872414, 173.024],
    145 : [1.3177235852557188, 4.4505895925855405, 173.024],
    146 : [1.3177235852557188, 4.4505895925855405, 173.024],
    147 : [1.3177235852557188, 4.4505895925855405, 173.024],
    148 : [1.3177235852557188, 4.4505895925855405, 173.024],
    151 : [1.3177235852557188, 4.974188368183839, 173.024],
    152 : [1.3177235852557188, 4.974188368183839, 173.024],
    153 : [1.3177235852557188, 4.974188368183839, 173.024],
    154 : [1.3177235852557188, 4.974188368183839, 173.024],
    157 : [1.3177235852557188, 5.497787143782138, 173.024],
    158 : [1.3177235852557188, 5.497787143782138, 173.024],
    159 : [1.3177235852557188, 5.497787143782138, 173.024],
    160 : [1.3177235852557188, 5.497787143782138, 173.024],
    163 : [1.3177235852557188, 6.021385919380437, 173.024],
    164 : [1.3177235852557188, 6.021385919380437, 173.024],
    165 : [1.3177235852557188, 6.021385919380437, 173.024],
    166 : [1.3177235852557188, 6.021385919380437, 173.024]
}


def cospsi(l1, l2):
    return (numpy.cos(locations[l1][0]) * numpy.cos(locations[l2][0]) + 
            numpy.sin(locations[l1][0]) * numpy.sin(locations[l2][0]) *
            numpy.cos(locations[l1][1] - locations[l2][1]))
 

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


def angular(dataset, gate_z, gate_y, gate_t=None, units=10, n_angles=100,
        chunk_size=10000000):
    """
    Calculate angular correlations distrubution between two Ge gates 
    """

    n = dataset.shape[0]
    left_pos = 0
    is_something_left = True
    t0 = datetime.datetime.now()
    progress_bar(left_pos, n)
    
    angles = numpy.zeros(n_angles)

    while is_something_left:
        right_pos = left_pos + chunk_size
        if right_pos > n:
            right_pos = n
            is_something_left = False
        submatrix = dataset[left_pos:right_pos, :]

        for loc in [[0, 1], [1, 0]]:
            logic = numpy.logical_and(
                                submatrix[:, loc[0]] >= gate_z[0] * units, 
                                submatrix[:, loc[0]] < gate_z[1] * units)
            logic = numpy.logical_and(logic,
                        numpy.logical_and(
                            submatrix[:, loc[1]] >= gate_y[0] * units, 
                            submatrix[:, loc[1]] < gate_y[1] * units))
            if gate_t is not None:
                logic = numpy.logical_and(logic, 
                            numpy.logical_and(
                                submatrix[:, loc[0]+2] >= gate_t[0] * 10, 
                                submatrix[:, loc[0]+2] < gate_t[1] * 10)
                            )
            logic = numpy.logical_not(logic)
            mask = numpy.repeat(logic, submatrix.shape[1])
            masked = numpy.ma.array(submatrix, mask=mask)
            selected = masked.compressed()
            selected = selected.reshape((int(selected.shape[0] / 6), 6))
            cos = []
            for event in selected:
                cos.append(cospsi(event[4], event[5]))
            bins, edges = numpy.histogram(cos, range=(-1, 1), bins=n_angles)
            angles += bins


        left_pos = right_pos
        dt = (datetime.datetime.now() - t0).total_seconds()
        progress_bar(left_pos, n, dt)

    print()
    print('Events in matrix:', angles.sum())
    return angles



def gege(dataset, gate_z, gate_t=None, 
           D=[4096], units=10, chunk_size=10000000):
    """
    Calculate Ge-Ge 1D matrix based on selected gate on Z axis 
    """

    n = dataset.shape[0]
    matrix = numpy.zeros(D)
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

        for loc in [[0, 1], [1, 0]]:
            logic = numpy.logical_and(
                                submatrix[:, loc[0]] >= gate_z[0] * units, 
                                submatrix[:, loc[0]] < gate_z[1] * units)
            if gate_t is not None:
                logic = numpy.logical_and(logic, 
                            numpy.logical_and(
                                submatrix[:, loc[0]+2] >= gate_t[0] * 10, 
                                submatrix[:, loc[0]+2] < gate_t[1] * 10)
                            )
            logic = numpy.logical_not(logic)
            mask = numpy.repeat(logic, submatrix.shape[1])
            masked = numpy.ma.array(submatrix, mask=mask)
            gg, edges = numpy.histogram(
                                 masked[:, loc[1]].compressed() / units,
                                 bins=D[0],
                                 range=(0, D[0]))
            matrix += gg

        left_pos = right_pos
        dt = (datetime.datetime.now() - t0).total_seconds()
        progress_bar(left_pos, n, dt)

    print()
    print('Events in matrix:', matrix.sum())
    return matrix


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NuBall 2D matrix gating script, on Ge-Ge events')
    parser.add_argument('-z', nargs=2, 
                        dest='gate_z', type=float, help='Z gate')
    parser.add_argument('-y', nargs=2, 
                        dest='gate_y', type=float, help='Y gate')
    parser.add_argument('-t', nargs=2, 
                        dest='gate_t', type=float, help='Time gate')
    parser.add_argument('--angular', action='store_true', 
            help='Use angular correlations, -z and -y must be set')
    parser.add_argument('--no_plot', action='store_true', 
                        help='Disable plotting')
    parser.add_argument('--save', nargs=1, 
                        help='Name of output text file')

    parser.add_argument('--chunk', type=int, default=10000000, 
            help='Size of data chunk loaded at time (default = 1e7), the larger the better, but reduce it if you run into memory issues')
    parser.add_argument('-D', type=int, nargs=1, default=[4096],
            help='Matrix dimension')
    parser.add_argument('-A', type=int, nargs=1, default=[100],
            help='Number of angular bins')

    parser.add_argument('input', help='Input HDF5 file')
    args = parser.parse_args()

    fin = h5py.File(args.input, 'r')

    dataset = fin['GG_list']
    print('# Number of entries', dataset.shape[0])

    n_angles = args.A[0]

    if not args.no_plot:
        plt.figure(1, (10, 6))
    if args.angular:
        if args.gate_z is None or args.gate_y is None:
            print('When using --angular, you must set -z and -y gate')
            exit()
        matrix = angular(dataset, args.gate_z, args.gate_y, args.gate_t, 
                n_angles=n_angles, chunk_size=args.chunk)
        if not args.no_plot:
            z_ax = numpy.linspace(-1, 1, n_angles + 1)[:-1]
            plt.xlabel(r'cos($\psi$)')
            plt.errorbar(z_ax, matrix, yerr=numpy.sqrt(matrix), marker='o', 
                    ls='None')
    else:
        if args.gate_z is None:
            print('When using Ge-Ge gate, you must use -z option')
            exit()
        matrix = gege(dataset, args.gate_z, args.gate_t, args.D,
            chunk_size=args.chunk)
        if not args.no_plot:
            z_ax = numpy.arange(matrix.shape[0])
            plt.xlabel('E (keV)')
            plt.plot(z_ax, matrix, ds='steps-mid')

    fin.close()

    if args.save is not None:
        header = 'File {},'.format(args.input)
        if args.gate_z is not None:
            header += ' gate z=[{0[0]}, {0[1]}],'.format(args.gate_z)
        if args.gate_y is not None:
            header += ' gate y=[{0[0]}, {0[1]}],'.format(args.gate_y)
        if args.gate_t is not None:
            header += ' gate t=[{0[0]}, {0[1]}],'.format(args.gate_t)
        numpy.savetxt(args.save[0], matrix, fmt="%d", header=header)


    if not args.no_plot:
        plt.ylim(0, None)
        plt.xlim(z_ax[0], z_ax[-1] * 1.1)
        plt.tight_layout()
        plt.show()

