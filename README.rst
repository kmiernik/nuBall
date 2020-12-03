===========
nuBall
===========

This package provides tools for nuBall HDF5 data files gating and
analysis. The main script is nucubes.py which calculates gamma-gamma-gamma
gates based on xml configuration file (see example file gates.xml)

The qtnuSpectra.py is a simple Qt-based GUI for quick data analysis. In 
works with gating result by nucubes.py saved to a HDF5 file.


Installation
------------

This package requires the following modules:
* matplotlib
* numpy
* h5py
* optional: numba for better performance

This software was tested on Linux only, installation on other OS may require
some manual intervention.  Installation module and scripts by pip is
recommended.

Warning: your distribution may use Python 2 as a default. If this is the
case use python3 / pip3 instead of python / pip command.

First create a source distribution

```
$ python setup.py sdist
```

Install it to for a local user

```
$ pip3 install .
```

or system-wide

```
$ sudo pip3 install .
```

The scripts (nucubes.py and qtnuSpectra.py) should be now available from
command line. If you chose local installation check if installation path
is in the PATH variable.
