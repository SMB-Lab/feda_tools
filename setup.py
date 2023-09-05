from setuptools import setup, find_packages

from feda_tools.__init__ import __version__

setup(
    name='feda_tools',
    version=__version__,

    url='https://github.com/SMB-Lab/feda_tools',
    author='Frank Duffy',
    author_email='fduffy0328@gmail.com',

    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        
    ]
)
