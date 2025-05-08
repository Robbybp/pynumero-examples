# PyNumero examples

This repository contains two examples of how PyNumero can be useful for:
1. Constructing the reduced Hessian and inspecting its eigenvalues
2. Inspecting the state trajectory during an optimization solve

This is not a beginner tutorial. If you don't know what PyNumero is,
see the PyNumero [documentation](https://pyomo.readthedocs.io/en/stable/explanation/solvers/pynumero/index.html),
which contains a basic quick-start guide.
Instead, these examples are intended to demonstrate two specific capabilities
that you can (relatively) easily apply to your own models.
Hopefully they are useful for you as well.

## Dependencies
These examples depend on [Pyomo](https://github.com/pyomo/pyomo),
the PyNumero-ASL library,
[IDAES](https://github.com/IDAES/idaes-pse), NumPy, SciPy, Matplotlib,
[CyIpopt](https://github.com/mechmotum/CyIpopt),
[Egret](https://github.com/grid-parity-exchange/Egret),
and this one random [repository](https://github.com/robbybp/surrogate-vs-implicit)
that I used for a paper at one point.
The full environment I used when developing the examples is in `requirements.txt`.
You can probably install it with:
```
pip install -r requirements.txt
```
The core functionality (that you need to apply this to your own models)
depends on Pyomo, PyNumero-ASL, NumPy, SciPy, and CyIpopt.
To build the PyNumero-ASL library after installing Pyomo, run:
```
pyomo build-extensions
```

## Citation
If you use PyNumero, please cite our paper!
```bibtex
@article{pynumero,
    author = {Rodriguez, Jose S. and Parker, Robert B. and Laird, Carl D. and Nicholson, Bethany L. and Siirola, John D. and Bynum, Michael L.},
    title = {Scalable Parallel Nonlinear Optimization with {PyNumero} and {Parapint}},
    journal = {INFORMS Journal on Computing},
    volume = {35},
    number = {2},
    pages = {509-517},
    year = {2023},
    doi = {10.1287/ijoc.2023.1272},
    eprint = {https://doi.org/10.1287/ijoc.2023.1272},
}
```

## LICENSE
This code is distributed under a BSD-3 license. See LICENSE.md.
