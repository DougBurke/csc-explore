#
# Allow the CSC test explorer to be installed
#
from setuptools import find_packages, setup

setup(
    name='csc-explorer',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask',
        'sherpa'  # really want CIAO but not really a package
    ],
)
