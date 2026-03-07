from setuptools import setup, find_packages

setup(
    name="aaa_tps",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "numba", "tqdm"],  # List dependencies here
    entry_points={},
)
