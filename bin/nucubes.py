#!/usr/bin/env python3

"""
Tools to make gates on ge-ge-ge, (ge-ge-la and ge-la-la not implemented in 
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

# TO DO:
# Load comment and add as a attribute (string)

import argparse
import h5py
import numpy
import datetime
import xml.dom.minidom

try:
    import numba
    from nuBall.nugates_numba import gegege
    print('Using numba version')
except ImportError:
    from nuBall.nugates import gegege


def save_txt(data, props):
    header = 'Data file {},'.format(props['data_file_name'])
    header += ' z = [{} {}]'.format(props['z'][0], props['z'][1])
    header += ' y = [{} {}]'.format(props['y'][0], props['y'][1])
    header += ' ' + props['detectors']
    header += ' ' + ''.join(props['gate_type'])

    z_mid = (props['z'][0] + props['z'][1]) / 2
    y_mid = (props['y'][0] + props['y'][1]) / 2
    output_name = '{}_{}_{}_{}_{}_{}'.format(props['target_alias'],
            props['isotope'], props['detectors'], ''.join(props['gate_type']),
            int(z_mid), int(y_mid))
    if len(props['m']) == 1:
        output_name += '_m{}'.format(props['m'][0])
    elif len(props['m']) == 2:
        output_name += '_m{}_{}'.format(props['m'][0], props['m'][1])
    output_name += '.txt'

    numpy.savetxt(output_name, data, fmt="%d", header=header)


def open_create_group(parent, group_name):
    try:
        group = parent['{}'.format(group_name)]
    except KeyError:
        group = parent.create_group('{}'.format(group_name))
    return group



def save_h5(data, out_name, props):
    try:
        fout = h5py.File(out_name, 'a', libver='latest')
    except OSError:
        print('Unable to open file', args.fout)
        return None

    target_group = open_create_group(fout, 
                                    '{}'.format(props['target_alias']))
    isotope_group = open_create_group(target_group, 
                                    '{}'.format(props['isotope']))
    detectors_group = open_create_group(isotope_group, 
                                    '{}'.format(props['detectors']))
    type_group = open_create_group(detectors_group, 
                                    '{}'.format(''.join(props['gate_type'])))
    data_name = '{}_{}'.format(int((props['z'][0] + props['z'][1]) / 2),
                               int((props['y'][0] + props['y'][1]) / 2))
    if len(props['m']) == 1:
        data_name += '_m{}'.format(props['m'][0])
    elif len(props['m']) == 2:
        data_name += '_m{}_{}'.format(props['m'][0], props['m'][1])

    i = 0
    while True:
        try:
            name = data_name + i * '*'
            type_group.create_dataset(name, data=data)
            type_group[name].attrs.create('z', props['z'])
            type_group[name].attrs.create('y', props['z'])

            M = [3, 9]
            if len(props['m']) == 0:
                mmin = M[0]
                mmax = M[-1]
            elif len(props['m']) == 1:
                mmin = max(props['m'][0], M[0])
                mmax = M[-1]
            else:
                mmax = min(props['m'][1], M[-1])
                mmin = max(props['m'][0], M[0])
            multi = [x for x in range(mmin, mmax+1)]
            type_group[name].attrs.create('m', multi)
        except ValueError:
            i += 1
            continue
        break
    fout.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NuBall 3D matrix gating script')
    parser.add_argument('config', help='Config XML file')
    parser.add_argument('-o', 
            help='Output HDF5 file (optional) if used result is save to HDF5 file, otherwise to a text file')
    args = parser.parse_args()
    props = {}
    try:

        dom = xml.dom.minidom.parse(args.config)
        config = dom.getElementsByTagName('config')[0]

        sections = config.getElementsByTagName('section')
        for section in sections:
            props['data_file_name'] = section.\
                    getElementsByTagName('data_file')[0].getAttribute('path')
            props['target_alias'] = section.\
                    getElementsByTagName('data_file')[0].getAttribute('alias')

            try:
                fin = h5py.File(props['data_file_name'], 'r')
            except OSError:
                print('Unable to open file', props['data_file_name'])
                continue

            time_windows = section.getElementsByTagName('time_windows')[0]

            # All tuples are for optimalization in numba mode
            # but do not affect 'normal' mode
            props['prompt'] = tuple(float(x) for x in 
                    time_windows.getAttribute('prompt').split())
            props['delayed'] = tuple(float(x) for x in 
                    time_windows.getAttribute('delayed').split())

            gates = section.getElementsByTagName('gate')
            for gate in gates:
                active = gate.getAttribute('active').lower() in ('true', 'yes')
                if not active:
                    continue
                props['isotope'] = gate.getAttribute('isotope')
                props['comment'] = gate.getAttribute('comment')

                props['z'] = tuple(float(x) for x in 
                                    gate.getAttribute('z').split())
                props['y'] = tuple(float(x) for x in 
                                    gate.getAttribute('y').split())


                props['detectors'] = gate.getAttribute('detectors')
                props['gate_type'] = tuple(x for x in gate.getAttribute('type'))
                props['m'] = tuple(int(x) for x in 
                                    gate.getAttribute('multi').split())

                print('* Processing gate', props['z'], props['y'],
                        props['detectors'], ''.join(props['gate_type']),
                        props['m'])
                data = gegege(fin, props['z'], props['y'], props['prompt'],
                        props['delayed'], props['m'], props['gate_type'])

                if args.o is None:
                    save_txt(data, props)
                else:
                    save_h5(data, args.o, props)


    except ValueError as err:
        print('Error parsing xml file:')

        print(err)
        exit()
