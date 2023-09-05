from setuptools import setup, find_packages

from feda_tools.__init__ import __version__

extra_dev = [
        'pytest>=4',
        'pytest-cov>=2',
    ]

setup(
    name='feda_tools',
    version=__version__,

    url='https://github.com/SMB-Lab/feda_tools',
    author='Frank Duffy',
    author_email='fduffy0328@gmail.com',

    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
    ],
    extras_require= {
        'dev': extra_dev,
    },
    entry_points={
    'console_scripts': [
        'cmd=feda_tools.twodim_hist:cmd_args',
        ],
    },
)
