from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

#python setup.py build_ext --inplace
filename = 'rc_region.pyx'

ext_modules = [Extension(filename.split('.')[0], 
              [filename],language='c++')]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[np.get_include()]
)