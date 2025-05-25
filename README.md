# Gaspype
The python package provides an performant library for thermodynamic calculations
like equilibrium reactions for several hundred gas species and their mixtures -
written in Python/Numpy.

Species are treated as ideal gases. Therefore the application is limited to moderate
pressures or high temperature applications.

Its designed with goal to be portable to Numpy-style GPU frameworks like JAX and PyTorch.

## Key features
- Tensor operations to prevent bottlenecks by the python interpreter
- Immutable types
- Elegant pythonic interface
- Great readable and compact syntax when using this package
- Good usability in Jupyter Notebook
- High performance for multidimensional fluid arrays

## Installation
Installation with pip:
``` bash
pip install gaspype
```

Installation with conda:
``` bash
conda install conda-forge::gaspype
```

## Getting started
Gaspype provides two main classes: ```fluid``` and ```elements```.

### Fluid
A fluid class describes a mixture of molecular species and their individual molar amounts.

``` python
import gaspype as gp
fl = gp.fluid({'H2O': 1, 'H2': 2})
fl
```
```
Total            3.000e+00 mol
H2O              33.33 %
H2               66.67 %
```

Its' functions provides thermodynamic, mass balance and ideal gas properties of the mixture.

``` python
cp = fl.get_cp(t=800+273.15)
mass = fl.get_mass()
gas_volume = fl.get_v(t=800+273.15, p=1e5)
```

The arguments can be provided as numpy-arrays:

``` python
import numpy as np
t_range = np.linspace(600, 800, 5) + 273.15
fl.get_density(t=t_range, p=1e5)
```
```
array([0.10122906, 0.09574625, 0.09082685, 0.08638827, 0.08236328])
```
A ```fluid``` object can have multiple compositions. A multidimensional ```fluid``` object can be created for example by multiplication with a numpy array:

``` python
fl2 = gp.fluid({'H2O': 1, 'N2': 2}) + \
      np.linspace(0, 10, 4) * gp.fluid({'H2': 1})
fl2
```
```
Total mol:
array([ 3.        ,  6.33333333,  9.66666667, 13.        ])
Species:
              H2        H2O         N2
Molar fractions:
array([[0.        , 0.33333333, 0.66666667],
       [0.52631579, 0.15789474, 0.31578947],
       [0.68965517, 0.10344828, 0.20689655],
       [0.76923077, 0.07692308, 0.15384615]])
```
A fluid object can be converted to a pandas dataframe:
``` python
import pandas as pd
pd.DataFrame(list(fl2))
```
|    | H2O | N2  |  H2 
|----|-----|-----|-------
|0   | 1.0 | 2.0 | 0.000000
|1   | 1.0 | 2.0 | 3.333333
|2   | 1.0 | 2.0 | 6.666667
|3   | 1.0 | 2.0 | 10.000000

The broadcasting behavior is not limited to 1D-arrays:

``` python
fl3 = gp.fluid({'H2O': 1}) + \
      np.linspace(0, 10, 4) * gp.fluid({'H2': 1}) + \
      np.expand_dims(np.linspace(1, 3, 3), axis=1) * gp.fluid({'N2': 1})
fl3
```
```
Total mol:
array([[ 2.        ,  5.33333333,  8.66666667, 12.        ],
       [ 3.        ,  6.33333333,  9.66666667, 13.        ],
       [ 4.        ,  7.33333333, 10.66666667, 14.        ]])
Species:
              H2        H2O         N2
Molar fractions:
array([[[0.        , 0.5       , 0.5       ],
        [0.625     , 0.1875    , 0.1875    ],
        [0.76923077, 0.11538462, 0.11538462],
        [0.83333333, 0.08333333, 0.08333333]],

       [[0.        , 0.33333333, 0.66666667],
        [0.52631579, 0.15789474, 0.31578947],
        [0.68965517, 0.10344828, 0.20689655],
        [0.76923077, 0.07692308, 0.15384615]],

       [[0.        , 0.25      , 0.75      ],
        [0.45454545, 0.13636364, 0.40909091],
        [0.625     , 0.09375   , 0.28125   ],
        [0.71428571, 0.07142857, 0.21428571]]])
```

### Elements
In some cases not the molecular but the atomic composition is of interest. The ```elements``` class can be used for atom based balances and works similar:

``` python
el = gp.elements({'N': 1, 'Cl': 2})
el.get_mass()
```
```
np.float64(0.08490700000000001)
```
A ```elements``` object can be as well instantiated from a ```fluid``` object. Arithmetic operations between ```elements``` and ```fluid``` result in an ```elements``` object:
``` python
el2 = gp.elements(fl) + el - 0.3 * fl
el2
```
```
Cl               2.000e+00 mol
H                4.200e+00 mol
N                1.000e+00 mol
O                7.000e-01 mol
```

Going from an atomic composition to an molecular composition is a little bit less straight forward, since there is no universal approach. One way is to calculate the thermodynamic equilibrium for a mixture:

``` python
fs = gp.fluid_system('CH4, H2, CO, CO2, O2')
el3 = gp.elements({'C': 1, 'H': 2, 'O':1}, fs)
fl3 = gp.equilibrium(el3, t=800)
fl3
```
```
Total            1.204e+00 mol
CH4              33.07 %
H2               16.93 %
CO               16.93 %
CO2              33.07 %
O2                0.00 %
```

The ```equilibrium``` function can be called with a ```fluid``` or ```elements``` object as first argument. ```fluid``` and ```elements``` referencing a ```fluid_system``` object witch can be be set as shown above during the object instantiation. If not provided, a new one will be created automatically. Providing a ```fluid_system``` gives more control over which molecular species are included in derived ```fluid``` objects. Furthermore arithmetic operations between objects with the same ```fluid_system``` are potentially faster:

``` python
fl3 + gp.fluid({'CH4': 1}, fs)
```
```
Total            2.204e+00 mol
CH4              63.44 %
H2                9.24 %
CO                9.24 %
CO2              18.07 %
O2                0.00 %
```

Especially if the ```fluid_system``` of one of the operants has not a subset of molecular species of the other ```fluid_system``` a new ```fluid_system``` will be created for the operation which might degrade performance:

``` python
fl3 + gp.fluid({'NH3': 1})
```
```
Total            2.204e+00 mol
CH4              18.07 %
CO                9.24 %
CO2              18.07 %
H2                9.24 %
NH3              45.38 %
O2                0.00 %
```

## Developer Guide
Contributions are welcome, please open an issue or submit a pull request on GitHub.

To get started with developing the `gaspype` package, follow these steps.

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/DLR-Institute-of-Future-Fuels/gaspype.git
cd gaspype
```

It's recommended to setup an venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the package and dev-dependencies while keeping the package files
in the current directory:

```bash
pip install -e .[dev]
```

Ensure that everything is set up correctly by running the tests:

```bash
pytest
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
