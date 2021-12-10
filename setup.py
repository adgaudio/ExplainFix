#!/usr/bin/env python
from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='explainfix',
    version='0.3.0',
    description='API for Explainable Spatially Fixed Deep Networks.',
    long_description=README,
    url="https://github.com/adgaudio/explainfix",
    author='Alex Gaudio',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    packages=['explainfix', 'explainfix.kernel', 'explainfix.models'],
    scripts=[],
    install_requires=[
        'efficientnet_pytorch', 'matplotlib', 'networkx', 'numpy',
        #  'torch', 'torchvision',
    ]
)
