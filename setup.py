from setuptools import setup, find_packages

from feda_tools.__init__ import __version__

extra_bin = [
    'numpy',
    'pandas',
    'matplotlib',

]

extra_test = [
    'pytest>=4',
    'pytest-cov>=2',
]

extra_dev = [
    *extra_test
]

extra_ci = [
    *extra_test,
    'python-coveralls',
]

setup(
    name='feda_tools',
    version=__version__,

    url='https://github.com/SMB-Lab/feda_tools',
    author='Frank Duffy',
    author_email='fduffy0328@gmail.com',

    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=[
        'tqdm==4.66.1',
        'numpy==1.26.2',
        'pandas==2.1.4',
        'scipy==1.11.4',
        'seaborn==0.13.0',
        'matplotlib==3.8.3',
        'pathlib==1.0.1',
        'tttrlib>=0.23.9',
        'pyyaml',
    ],
    python_requires= '>=3.11.5, <=3.11.9',
    extras_require= {
        'dev' : extra_dev,
        'ci' : extra_ci,
    },
    entry_points={
    'console_scripts': [
        '2dhist=feda_tools.twodim_hist:make_2dhist',
        ],
    },
)
