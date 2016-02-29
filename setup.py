from setuptools import setup, find_packages
import os,sys

def readme():
    with open('README.rst') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__EXOSYSPOP_SETUP__ = True
import exosyspop
version = exosyspop.__version__

setup(name = "exosyspop",
    version = version,
    description = "Monte Carlo Simulations of Transit/Eclipse Surveys",
    long_description = readme(),
    author = "Timothy D. Morton",
    author_email = "tim.morton@gmail.com",
    url = "https://github.com/timothydmorton/exosyspop",
    packages = find_packages(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    install_requires=['isochrones>=0.9.0', 'vespa>=0.4.1'],
    zip_safe=False
) 
