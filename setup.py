from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cvxstrat",
    version="1.0.0",
    author="Jonathan Tuck, Shane Barratt",
    description="Laplacian regularized stratified models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.15",
        "scipy >= 1.2",
        "networkx >= 2.2",
        "scikit-learn >= 0.20",
        "torch >= 1.0"],
    license="Apache License, Version 2.0",
    url="http://github.com/cvxgrp/cvxstrat/",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
