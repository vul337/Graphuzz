#!/usr/bin/env python
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize

    ext_modules = cythonize([
        Extension("cy.exportTrainingSet", ["cy/exportTrainingSet.pyx"]),
        Extension("cy.nn_server", ["cy/nn_server.pyx"]),
    ])
except ImportError:
    __import__("traceback").print_exc()
    ext_modules = None

setup(
    ext_modules=ext_modules
)
