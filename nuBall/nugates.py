"""
Functions to make gates on ge-ge-ge, (ge-ge-la and ge-la-la not implemented in 
this version yet) coincidences
Input file must be organized as follow:
    HDF5 dataset GGG_list
    group for each multiplicity 3, 4, 5, ...

    matrix n * 7, STD_U32LE

    E0 E1 E2 T0 T1 T2 pattern
    
    where pattern is mulitplicity * 100 + hit_pattern

    This version is for multiplicity and open coin window
    This version uses numpy smart indexing instead of logic masks
"""

import datetime
import numpy
from nuBall.tools import progress_bar


def gegege(fin, gate_z, gate_y, prompt, delayed, gate_m, gate_type,
           D=4096, E_unit=100, t_unit=1000, chunk_size=10000000):
    """
    Calculate Ge-Ge-Ge 1D matrix based on selected gate on Z axis 
     and Y-axis.
    * fin is HDF5 file with proper structure
    * gate_z, gate_y are gates (floating point numbers)
    * prompt should be tuple [t0, t1] defining range for prompt gamma
      e.g. [27, 60]
    * delayed defines range for delayed gamma (as above)
    * gate_m is multiplicity gate, if m is [], whole range is used,
      if len(m) == 1, it is understood as lower limit
    * gate_type is three letter string, where 'p' stands for prompt, 
      'd' for delayed, and any other letter for all; each letter is 
      for subsequent detector
    * D is dimension of array (1 keV per channel)
    * E_unit is number by which that gates should be multiplied in order
      to have same units as in data file (if it is e.g. 10 eV than E_unit
      is 100, so the numbers are given in keV)
    * t_unit is for time, as above
    * chunk_size is the chunk of input file loaded at a time
    """

    group = fin['GeGeGe']
    M = []
    for key in group.keys():
        M.append(int(key))

    if len(gate_m) == 0:
        multi = M
    elif len(gate_m) == 1:
        multi = [x for x in range(gate_m[0], M[-1]+1)]
    else:
        mmin = max(gate_m[0], M[0])
        mmax = min(gate_m[1], M[-1])
        multi = [x for x in range(gate_m[0], mmax+1)]
    matrix = numpy.zeros(D)

    n_all = 0
    n_processed = 0
    for i, m in enumerate(multi):
        dataset = group['{}'.format(m)] 
        n_all += dataset.shape[0]

    t0 = datetime.datetime.now()
    progress_bar(0, n_all)
    for m in multi:
        dataset = group['{}'.format(m)] 
        n = dataset.shape[0]
        left_pos = 0
        is_something_left = True
        while is_something_left:
            right_pos = left_pos + chunk_size
            if right_pos > n:
                right_pos = n
                is_something_left = False
            submatrix = numpy.array(dataset[left_pos:right_pos, :])

            for loc in [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], 
                    [2, 0, 1], [2, 1, 0]]:
                sel = submatrix[submatrix[:, loc[0]] >= gate_z[0] * E_unit]
                sel = sel[sel[:, loc[0]] < gate_z[1] * E_unit]
                sel = sel[sel[:, loc[1]] >= gate_y[0] * E_unit]
                sel = sel[sel[:, loc[1]] < gate_y[1] * E_unit]
                for i, t in enumerate(gate_type):
                    if t == 'p':
                        sel = sel[sel[:, loc[i]+3] >= prompt[0] * t_unit]
                        sel = sel[sel[:, loc[i]+3] <= prompt[1] * t_unit]
                    elif t == 'd':
                        sel = sel[sel[:, loc[i]+3] >= delayed[0] * t_unit]
                        sel = sel[sel[:, loc[i]+3] <= delayed[1] * t_unit]

                gg, edges = numpy.histogram( sel[:, loc[2]] / E_unit, 
                                            bins=D, range=[0, D])
                matrix += gg
            n_processed += right_pos - left_pos
            left_pos = right_pos
            dt = (datetime.datetime.now() - t0).total_seconds()
            progress_bar(n_processed, n_all, dt)
    print()
    print('Events in matrix:', matrix.sum())
    return matrix


if __name__ == '__main__':
    pass

