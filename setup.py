#src/shared_logic/setup.py
import os

from setuptools import setup, find_packages

version = os.getenv("PACKAGE_VERSION", "0.0.1")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='hdsemg-shared',
    version=version,
    author="Johannes Kasser",
    author_email="johanneskasser@outlook.de",
    description="Utility Methods for hd-semg files.",
    url="https://github.com/johanneskasser/hdsemg-shared.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    python_requires='>=3.8',
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)