# Gmsh for FEniCS
Mesh the model and load the result (Mesh, MeshFunction) in memory (serial only).


# Installation
For dev mode `pip install -e .` from folder where `README.md` resides. 
Test things by running `py.test .` We have the following dependencies

- Gmsh (tested against version 4.8.4)
- FEniCS (tested against version 2019.1.0)
- tqdm

Python version tested with the above is `3.9.6`

## TODO
- [ ] Mesh and load in parallel
- [ ] Consider bundling MeshFunctions as MeshData into mesh?