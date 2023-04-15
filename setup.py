"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages

# Package meta-data.
NAME = "wfc_reference"
DESCRIPTION = "TBD"
URL = "https://github.com/NREL/wfc_reference"
EMAIL = "paul.fleming@nrel.gov"
AUTHOR = "NREL National Wind Technology Center"

# What packages are required for this module to be executed?
REQUIRED = [
    'feather-format',
    'matplotlib>=3.6.3',
    'numpy',
    'scipy',
    'pandas'
]

ROOT = Path(__file__).parent
with open(ROOT / "wfc_reference" / "version.py") as version_file:
    VERSION = version_file.read().strip()

with open('README.rst') as readme_file:
    README = readme_file.read()

setup_requirements = [
    # Placeholder
]

test_requirements = [
    # Placeholder
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(include=['wfc_reference']),
    # entry_points={
    #     'console_scripts': [
    #         'flasc=flasc.cli:main'
    #     ]
    # },
    include_package_data=True,
    install_requires=REQUIRED,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='wfc_reference',
    classifiers=[
        'Development Status :: Release',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
