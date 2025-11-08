from setuptools import setup, find_packages

setup(
    name="my-transformer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.5",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.3",
    ],
)