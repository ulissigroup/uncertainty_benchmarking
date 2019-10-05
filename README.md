This reposity should be used as a sandbox for performing various ML regressions with uncertainties.
To start, we are modeling CO adsorption energy data we get from GASpy.

# Dependencies

Regardless of where you try to run these notebooks, you'll probably need to clone the `sklearn-refactor` branch of `https://github.com/ulissigroup/cgcnn` and add it to your `$PYTHONPATH`.

## Cori

Note that on Cori, you should update your `$PYTHONPATH` in `~/.bashrc.ext`, not `~/.bashrc`.

If you want to pull the data in a live fashion (i.e., use `gaspy.gasdb`), you should add this to your `~/.bashrc.ext` as well.

    export PYTHONPATH="/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/:${PYTHONPATH}"
    export PYTHONPATH="/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_feedback/:${PYTHONPATH}"
    export PYTHONPATH="/global/project/projectdirs/m2755/GASpy_workspaces/GASpy/GASpy_regressions/:${PYTHONPATH}"

If you have access to the Cori GPUs, then you should create this file in `~/.local/share/jupyter/kernels/gaspy_ktran/kernel.json`:

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
To use it, you can you visit `https://jupyter.nersc.gov/hub/home` and then start up an "Exclusive GPU Node".
When you open a notebook, choose the `gaspy_ktran` kernel.
Now you're ready to roll.

## Make-your-own

I'm actually not sure what dependencies are all needed to run all the notebooks here.
I know you need [ASE](https://anaconda.org/conda-forge/ase), [pymatgen](https://anaconda.org/matsci/pymatgen), PyTorch, GPyTorch, SKLearn, Jupyter, seaborn, shapely, etc., but I'm not positive if that's all-encompassing.
What I do remember is that you need to hack your installation of `skorch`. In `skorch/skorch/dataset.py`, delete these lines:

    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")

Otherwise, here is what `conda list --revisions` gives me (if that helps):

    2019-08-06 16:08:58  (rev 0)

    2019-08-07 08:58:35  (rev 1)
        +_libgcc_mutex-0.1
        +apscheduler-3.6.1 (conda-forge)
        +ase-3.18.0 (conda-forge)
        +asn1crypto-0.24.0 (conda-forge)
        +atomicwrites-1.3.0 (conda-forge)
        +attrs-19.1.0 (conda-forge)
        +backcall-0.1.0 (conda-forge)
        +blas-1.0
        +bleach-3.1.0 (conda-forge)
        +bzip2-1.0.8 (conda-forge)
        +ca-certificates-2019.6.16 (conda-forge)
        +certifi-2019.6.16 (conda-forge)
        +cffi-1.12.3 (conda-forge)
        +cftime-1.0.3.4 (conda-forge)
        +chardet-3.0.4 (conda-forge)
        +click-7.0 (conda-forge)
        +colorama-0.4.1 (conda-forge)
        +cryptography-2.7 (conda-forge)
        +cudatoolkit-10.0.130
        +cudnn-7.6.0
        +curl-7.65.3 (conda-forge)
        +cycler-0.10.0 (conda-forge)
        +dataclasses-0.6 (conda-forge)
        +dbus-1.13.6 (conda-forge)
        +deap-1.3.0 (conda-forge)
        +decorator-4.4.0 (conda-forge)
        +defusedxml-0.5.0 (conda-forge)
        +dill-0.3.0 (conda-forge)
        +entrypoints-0.3 (conda-forge)
        +expat-2.2.5 (conda-forge)
        +fastcache-1.1.0 (conda-forge)
        +fireworks-1.7.2 (matsci)
        +flask-1.1.1 (conda-forge)
        +flask-paginate-0.5.1 (matsci)
        +fontconfig-2.13.1 (conda-forge)
        +freetype-2.10.0 (conda-forge)
        +future-0.17.1 (conda-forge)
        +gettext-0.19.8.1 (conda-forge)
        +glib-2.58.3 (conda-forge)
        +gmp-6.1.2 (conda-forge)
        +gmpy2-2.1.0b1 (conda-forge)
        +gpytorch-0.3.4 (gpytorch)
        +gst-plugins-base-1.14.5 (conda-forge)
        +gstreamer-1.14.5 (conda-forge)
        +gunicorn-19.9.0 (conda-forge)
        +hdf4-4.2.13 (conda-forge)
        +hdf5-1.10.5 (conda-forge)
        +icu-64.2 (conda-forge)
        +idna-2.8 (conda-forge)
        +importlib_metadata-0.18 (conda-forge)
        +intel-openmp-2019.4
        +ipykernel-5.1.1 (conda-forge)
        +ipython-7.7.0 (conda-forge)
        +ipython_genutils-0.2.0 (conda-forge)
        +ipywidgets-7.5.1 (conda-forge)
        +itsdangerous-1.1.0 (conda-forge)
        +jedi-0.14.1 (conda-forge)
        +jinja2-2.10.1 (conda-forge)
        +joblib-0.13.2 (conda-forge)
        +jpeg-9c (conda-forge)
        +jsoncpp-1.8.4 (conda-forge)
        +jsonschema-3.0.2 (conda-forge)
        +jupyter-1.0.0 (conda-forge)
        +jupyter_client-5.3.1 (conda-forge)
        +jupyter_console-6.0.0 (conda-forge)
        +jupyter_contrib_core-0.3.3 (conda-forge)
        +jupyter_core-4.4.0 (conda-forge)
        +jupyter_nbextensions_configurator-0.4.1 (conda-forge)
        +kiwisolver-1.1.0 (conda-forge)
        +krb5-1.16.3 (conda-forge)
        +latexcodec-1.0.7 (conda-forge)
        +libblas-3.8.0 (conda-forge)
        +libcblas-3.8.0 (conda-forge)
        +libcurl-7.65.3 (conda-forge)
        +libedit-3.1.20181209
        +libffi-3.2.1 (conda-forge)
        +libgcc-ng-9.1.0
        +libgfortran-ng-7.3.0
        +libiconv-1.15 (conda-forge)
        +liblapack-3.8.0 (conda-forge)
        +libnetcdf-4.6.2 (conda-forge)
        +libpng-1.6.37 (conda-forge)
        +libsodium-1.0.17 (conda-forge)
        +libssh2-1.8.2 (conda-forge)
        +libstdcxx-ng-9.1.0
        +libtiff-4.0.10 (conda-forge)
        +libuuid-2.32.1 (conda-forge)
        +libxcb-1.13 (conda-forge)
        +libxml2-2.9.9 (conda-forge)
        +lockfile-0.12.2 (conda-forge)
        +luigi-2.1.0 (conda-forge)
        +lz4-c-1.8.3 (conda-forge)
        +markupsafe-1.1.1 (conda-forge)
        +matplotlib-3.1.1 (conda-forge)
        +matplotlib-base-3.1.1 (conda-forge)
        +mendeleev-0.4.5 (conda-forge)
        +mistune-0.8.4 (conda-forge)
        +mkl-2019.4
        +mongodb-4.0.3
        +monty-2.0.4 (conda-forge)
        +more-itertools-7.2.0 (conda-forge)
        +mpc-1.1.0 (conda-forge)
        +mpfr-4.0.2 (conda-forge)
        +mpmath-1.1.0 (conda-forge)
        +multiprocess-0.70.8 (conda-forge)
        +nbconvert-5.5.0 (conda-forge)
        +nbformat-4.4.0 (conda-forge)
        +ncurses-6.1 (conda-forge)
        +netcdf4-1.5.1.2 (conda-forge)
        +networkx-2.3 (conda-forge)
        +ninja-1.9.0 (conda-forge)
        +notebook-5.7.8 (conda-forge)
        +numpy-1.17.0 (conda-forge)
        +olefile-0.46 (conda-forge)
        +openssl-1.1.1c (conda-forge)
        +packaging-19.0 (conda-forge)
        +palettable-3.2.0 (conda-forge)
        +pandas-0.25.0 (conda-forge)
        +pandoc-2.7.3 (conda-forge)
        +pandocfilters-1.4.2 (conda-forge)
        +parso-0.5.1 (conda-forge)
        +patsy-0.5.1 (conda-forge)
        +pcre-8.41 (conda-forge)
        +pexpect-4.7.0 (conda-forge)
        +pickleshare-0.7.5 (conda-forge)
        +pillow-6.1.0 (conda-forge)
        +pip-19.2.1 (conda-forge)
        +plotly-4.1.0 (conda-forge)
        +pluggy-0.12.0 (conda-forge)
        +prometheus_client-0.7.1 (conda-forge)
        +prompt_toolkit-2.0.9 (conda-forge)
        +pthread-stubs-0.4 (conda-forge)
        +ptyprocess-0.6.0 (conda-forge)
        +py-1.8.0 (conda-forge)
        +pybtex-0.22.2 (conda-forge)
        +pycparser-2.19 (conda-forge)
        +pydispatcher-2.0.5 (conda-forge)
        +pyfiglet-0.8.post1 (conda-forge)
        +pygments-2.4.2 (conda-forge)
        +pymatgen-2019.7.30 (conda-forge)
        +pymongo-3.8.0 (conda-forge)
        +pyopenssl-19.0.0 (conda-forge)
        +pyparsing-2.4.2 (conda-forge)
        +pyqt-5.9.2 (conda-forge)
        +pyrsistent-0.15.4 (conda-forge)
        +pysocks-1.7.0 (conda-forge)
        +pytest-5.0.1 (conda-forge)
        +python-3.6.7 (conda-forge)
        +python-daemon-2.2.3 (conda-forge)
        +python-dateutil-2.8.0 (conda-forge)
        +pytorch-1.0.1
        +pytz-2019.2 (conda-forge)
        +pyyaml-5.1.2 (conda-forge)
        +pyzmq-18.0.2 (conda-forge)
        +qt-5.9.7 (conda-forge)
        +qtconsole-4.5.2 (conda-forge)
        +readline-7.0 (conda-forge)
        +requests-2.22.0 (conda-forge)
        +retrying-1.3.3 (conda-forge)
        +ruamel-1.0 (conda-forge)
        +ruamel.yaml-0.16.0 (conda-forge)
        +scikit-learn-0.21.3 (conda-forge)
        +scipy-1.3.0 (conda-forge)
        +seaborn-0.9.0 (conda-forge)
        +send2trash-1.5.0 (conda-forge)
        +setuptools-41.0.1 (conda-forge)
        +sip-4.19.8 (conda-forge)
        +six-1.12.0 (conda-forge)
        +skorch-0.6.0 (conda-forge)
        +spglib-1.14.1 (conda-forge)
        +sqlalchemy-1.3.6 (conda-forge)
        +sqlite-3.29.0
        +statsmodels-0.10.1 (conda-forge)
        +stopit-1.1.2 (conda-forge)
        +sympy-1.4 (conda-forge)
        +tabulate-0.8.3 (conda-forge)
        +tbb-2019.7 (conda-forge)
        +terminado-0.8.2 (conda-forge)
        +testpath-0.4.2 (conda-forge)
        +tk-8.6.9 (conda-forge)
        +torchvision-0.2.1 (conda-forge)
        +tornado-4.5.3 (conda-forge)
        +tpot-0.10.2 (conda-forge)
        +tqdm-4.32.2 (conda-forge)
        +traitlets-4.3.2 (conda-forge)
        +tzlocal-2.0.0 (conda-forge)
        +update_checker-0.16 (conda-forge)
        +urllib3-1.25.3 (conda-forge)
        +vtk-8.2.0 (conda-forge)
        +wcwidth-0.1.7 (conda-forge)
        +webencodings-0.5.1 (conda-forge)
        +werkzeug-0.15.5 (conda-forge)
        +wheel-0.33.4 (conda-forge)
        +widgetsnbextension-3.5.1 (conda-forge)
        +xorg-kbproto-1.0.7 (conda-forge)
        +xorg-libice-1.0.10 (conda-forge)
        +xorg-libsm-1.2.3 (conda-forge)
        +xorg-libx11-1.6.8 (conda-forge)
        +xorg-libxau-1.0.9 (conda-forge)
        +xorg-libxdmcp-1.1.3 (conda-forge)
        +xorg-libxt-1.2.0 (conda-forge)
        +xorg-xproto-7.0.31 (conda-forge)
        +xz-5.2.4 (conda-forge)
        +yaml-0.1.7 (conda-forge)
        +zeromq-4.3.2 (conda-forge)
        +zipp-0.5.2 (conda-forge)
        +zlib-1.2.11 (conda-forge)
        +zstd-1.4.0 (conda-forge)

    2019-08-07 09:12:45  (rev 2)
         hdf5  {1.10.5 (conda-forge) -> 1.10.4 (conda-forge)}
         libnetcdf  {4.6.2 (conda-forge) -> 4.6.2 (conda-forge)}
         netcdf4  {1.5.1.2 (conda-forge) -> 1.5.1.2 (conda-forge)}
         vtk  {8.2.0 (conda-forge) -> 8.2.0 (conda-forge)}
        +_tflow_select-2.3.0
        +absl-py-0.7.1 (conda-forge)
        +astor-0.7.1 (conda-forge)
        +c-ares-1.15.0 (conda-forge)
        +gast-0.2.2 (conda-forge)
        +google-pasta-0.1.7 (conda-forge)
        +grpcio-1.16.1
        +h5py-2.9.0 (conda-forge)
        +keras-applications-1.0.7 (conda-forge)
        +keras-preprocessing-1.0.9 (conda-forge)
        +libprotobuf-3.9.1 (conda-forge)
        +markdown-3.1.1 (conda-forge)
        +protobuf-3.9.1 (conda-forge)
        +tensorboard-1.14.0 (conda-forge)
        +tensorflow-1.14.0
        +tensorflow-base-1.14.0
        +tensorflow-estimator-1.14.0 (conda-forge)
        +termcolor-1.1.0 (conda-forge)
        +wrapt-1.11.2 (conda-forge)

    2019-08-08 09:41:46  (rev 3)
         _tflow_select  {2.3.0 -> 2.1.0}
         cudatoolkit  {10.0.130 -> 10.1.168}
         cudnn  {7.6.0 -> 7.6.0}
         pytorch  {1.0.1 -> 1.0.0 (pytorch)}
         tensorflow  {1.14.0 -> 1.14.0}
         tensorflow-base  {1.14.0 -> 1.14.0}
        +cupti-10.1.168
        +tensorflow-gpu-1.14.0

    2019-08-08 09:43:56  (rev 4)
        +binutils_impl_linux-64-2.31.1
        +binutils_linux-64-2.31.1
        +gcc_impl_linux-64-7.3.0 (conda-forge)
        +gcc_linux-64-7.3.0 (conda-forge)
        +gxx_impl_linux-64-7.3.0 (conda-forge)
        +gxx_linux-64-7.3.0 (conda-forge)
        +keras-2.2.4 (conda-forge)
        +libgpuarray-0.7.6 (conda-forge)
        +mako-1.1.0 (conda-forge)
        +pygpu-0.7.6 (conda-forge)
        +theano-1.0.4 (conda-forge)

    2019-08-08 09:49:09  (rev 5)
         h5py  {2.9.0 (conda-forge) -> 2.9.0 (conda-forge)}
         hdf5  {1.10.4 (conda-forge) -> 1.10.5 (conda-forge)}
         ipykernel  {5.1.1 (conda-forge) -> 5.1.2 (conda-forge)}
         libblas  {3.8.0 (conda-forge) -> 3.8.0 (conda-forge)}
         libcblas  {3.8.0 (conda-forge) -> 3.8.0 (conda-forge)}
         liblapack  {3.8.0 (conda-forge) -> 3.8.0 (conda-forge)}
         libnetcdf  {4.6.2 (conda-forge) -> 4.6.2 (conda-forge)}
         libxml2  {2.9.9 (conda-forge) -> 2.9.9 (conda-forge)}
         netcdf4  {1.5.1.2 (conda-forge) -> 1.5.1.2 (conda-forge)}
         ruamel.yaml  {0.16.0 (conda-forge) -> 0.16.1 (conda-forge)}
         vtk  {8.2.0 (conda-forge) -> 8.2.0 (conda-forge)}
        -blas-1.0
        +libopenblas-0.3.6 (conda-forge)

    2019-08-08 10:14:48  (rev 6)
         cudatoolkit  {10.1.168 -> 10.0.130}
         cudnn  {7.6.0 -> 7.6.0}
         cupti  {10.1.168 -> 10.0.130}
         pytorch  {1.0.0 (pytorch) -> 1.1.0 (pytorch)}
         tensorflow  {1.14.0 -> 1.14.0}
         tensorflow-base  {1.14.0 -> 1.14.0}

    2019-08-09 09:22:09  (rev 7)
        +flake8-3.7.8 (conda-forge)
        +mccabe-0.6.1 (conda-forge)
        +pycodestyle-2.5.0 (conda-forge)
        +pyflakes-2.1.1 (conda-forge)

    2019-08-09 14:47:08  (rev 8)
        +_py-xgboost-mutex-2.0 (conda-forge)
        +libxgboost-0.90 (conda-forge)
        +py-xgboost-0.90 (conda-forge)
        +xgboost-0.90 (conda-forge)
