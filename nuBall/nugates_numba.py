"""
Functions to make gates on ge-ge-ge, ge-ge-la and ge-la-la coincidences
Input file must be organized as follow:
    HDF5 dataset GGG_list
    matrix n * 7, STD_U32LE

    E0 E1 E2 T0 T1 T2 pattern
    
    where pattern is mulitplicity * 100 + hit_pattern

    This version is for multiplicity and open coin window
    This version uses numpy smart indexing instead of logic masks
    This version uses numba
"""

import datetime
import numpy
from numba import jit, prange
from nuBall.tools import progress_bar


@jit(nopython=True, parallel=True, fastmath=True)
def select(submatrix, matrix, gate_z, gate_y, prompt, delayed, gate_type,
           D, E_unit, t_unit):
    """
    This is version of gating tailored for numba parallel
    instead of smart numpy smart indexing it iterates by
    prange and checks the condition by stack of ifs
    """
    permutations = [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], 
                    [2, 0, 1], [2, 1, 0]]
    for k in prange(6):
        loc = permutations[k]
        n = submatrix.shape[0]
        gg = numpy.zeros(D)
        for i in prange(n):
            if submatrix[i, loc[0]] < gate_z[0] * E_unit:
                continue
            if submatrix[i, loc[0]] >= gate_z[1] * E_unit:
                continue
            if submatrix[i, loc[1]] < gate_y[0] * E_unit:
                continue
            if submatrix[i, loc[1]] >= gate_y[1] * E_unit:
                continue
            for j in prange(3):
                t = gate_type[j]
                if t == 'p':
                    if submatrix[i, loc[j] + 3] < prompt[0] * t_unit:
                        break
                    if submatrix[i, loc[j] + 3] > prompt[1] * t_unit:
                        break
                elif t == 'd':
                    if submatrix[i, loc[j] + 3] < delayed[0] * t_unit:
                        break
                    if submatrix[i, loc[j] + 3] > delayed[1] * t_unit:
                        break
            else:
                e = int(submatrix[i, loc[2]] / E_unit)
                if 0 <= e < D:
                    gg[e] += 1
        matrix += gg


def gegege(fin, gate_z, gate_y, prompt, delayed, gate_m, gate_type,
           D=4096, E_unit=100, t_unit=1000):
    """
    This version uses select function tailored for numba
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
        chunk_size = dataset.chunks[0]
        left_pos = 0
        is_something_left = True
        while is_something_left:
            right_pos = left_pos + chunk_size
            if right_pos > n:
                right_pos = n
                is_something_left = False
            submatrix = numpy.array(dataset[left_pos:right_pos, :])
            select(submatrix, matrix, gate_z, gate_y, prompt, delayed,
                    gate_type, D, E_unit, t_unit)

            n_processed += right_pos - left_pos
            left_pos = right_pos
            dt = (datetime.datetime.now() - t0).total_seconds()
            progress_bar(n_processed, n_all, dt)
    print()
    print('Events in matrix:', matrix.sum())
    return matrix



if __name__ == '__main__':
    pass
