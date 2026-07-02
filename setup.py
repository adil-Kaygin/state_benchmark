from setuptools import setup, find_packages

setup(
    name="state_benchmark",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "h5py>=3.8",
        "torch>=2.0",
        "matplotlib>=3.7",
        "comet_ml>=3.0",
        "torch-kf",
        "torchfilter @ git+https://github.com/stanford-iprl-lab/torchfilter.git",    ],
    extras_require={

        "fast": [
            "numba>=0.58",
        ],
    },
)
