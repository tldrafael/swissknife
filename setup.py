from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Swiss Knife'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name='swissknife',
        version=VERSION,
        author='Rafael de Souza Toledo',
        author_email='rafaeltol@gmail.com',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['torch', 'torchvision', 'numpy'],
        keywords=['python']
)
