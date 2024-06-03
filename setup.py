from setuptools import find_packages, setup

setup(
    name="data",
    version="0.1.0",
    author="Mehdi Azabou",
    author_email="mazabou@gatech.edu",
    description="A utility package for complex temporal data manipulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy~=1.23.5",
        "pandas~=1.5.3",
        "h5py~=3.8.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
