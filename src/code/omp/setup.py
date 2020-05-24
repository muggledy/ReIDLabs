import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#编译：python setup.py build_ext --inplace
filename='batch_omp.pyx' #.pyx源文件名

setup(
    name=filename.split('.')[0],
    cmdclass={'build_ext':build_ext},
    ext_modules=[Extension(filename.split('.')[0],sources=[filename,"main.cpp"],language='c++',include_dirs=[numpy.get_include()])],
)