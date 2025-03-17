# DNAFold-LAMMPS

See the repository for the original C++ implementation of DNAfold [here](https://github.com/daniel-duke/DNAfold).

See the documentation for DNAfold [here](https://daniel-duke.github.io/DNAfold-docs/).

Use the main script (`dnafold_lmp.py`) to generate the input files for running DNAfold in LAMMPS. Note that for the simulation to run, LAMMPS must be compiled with the react package.

When the simulation has finished, use the `backend_basics.py` script to process the data (calculate hybridization times, write various centered trajectories, etc). After this, the other simulation analysis scripts (`hyb_times_visual.py`, `crystallinity.py`, etc) can be run.
