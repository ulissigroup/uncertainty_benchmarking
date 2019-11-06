# Dependencies

You will need:
- Python >= 3.6.7
- [ASE](https://anaconda.org/conda-forge/ase) = 3.17
- [pymatgen](https://anaconda.org/matsci/pymatgen) = 2019.7.30
- Tensorflow-gpu >= 1.14
- PyTorch >= 1.1.0
- [CGCNN](https://github.com/ulissigroup/cgcnn/tree/sklearn_refactor)
- [GPyTorch](https://github.com/cornellius-gp/gpytorch) >= 0.3
- [Pyro](https://github.com/pyro-ppl/pyro) >= 0.5
- SKLearn >= 0.21
- Skorch >= 0.6
- Jupyter >= 1.0
- ipycache = 0.1.5.dev0
- Seaborn >= 0.9
- Shapely >= 1.6

You will also need to hack your installation of `skorch`.
In `skorch/skorch/dataset.py`, delete these lines:

    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")

This will let you run all of the modeling notebooks.
To run the notebooks that pull GASpy data (nicknamed "GASdb"), you will need to be a Ulissi group member on Cori with the right environment setup (see below).
Otherwise, you can simply use the pickle caches we have created in this repository.

# Ulissi group usage on Cori

For Ulissi group members planning to use Cori, you need to clone the `sklearn-refactor` branch of `https://github.com/ulissigroup/cgcnn` and add it to your `$PYTHONPATH`.
Remember to update your `$PYTHONPATH` in `~/.bashrc.ext`, not `~/.bashrc`.

Then you should create this file in `~/.local/share/jupyter/kernels/gaspy_ktran/kernel.json`:

    {
        "argv": [
            "/global/homes/k/ktran/miniconda3/envs/gaspy/bin/python",
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}"
        ],
        "display_name": "gaspy_ktran",
        "language": "python"
    }

This will give you access to Kevin's personal conda on Cori, which has everything else installed.
To use it, visit `https://jupyter.nersc.gov/hub/home` and select the `gaspy_ktran` kernel.
If you want to use a neural network and have access to NERSC GPUs, then you should select "Exclusive GPU Node".

If you want to pull the data in a live fashion (i.e., use `gaspy.gasdb`), you should add this to your `~/.bashrc.ext` as well.

    export PYTHONPATH="/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/:${PYTHONPATH}"
    export PYTHONPATH="/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_feedback/:${PYTHONPATH}"
    export PYTHONPATH="/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/:${PYTHONPATH}"

