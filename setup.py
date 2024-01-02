#!/usr/bin/env python
try:
    from numpy.distutils.core import Extension as NumpyExtension
    from numpy.distutils.core import setup

    from distutils.extension import Extension

    import numpy

except ImportError:
    raise ImportError('Numpy needs to be installed or updated.')

setup(
    name="seabinv",
    version="1.0",
    author="Cheng Song",
    author_email="songcheng@snu.ac.kr",
    description=("Parallel Tempering Transdimensional Bayesian Inversion of Relative Sea level"),
    install_requires=[],
    url="",
    packages=['seabinv'],
    package_dir={
        'seabinv': 'src'},
    scripts=['src/scripts/baywatch_rsl'],
    package_data={
        'seabinv': ['defaults/*'], },
)
