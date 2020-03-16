import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TheSequencer",
    version="0.0.1",
    description="An algorithm that detects one-dimensional sequences in complex datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    keywords="dimensionality reduction t-sne UMAP",
    url="https://github.com/dalya/Sequencer",
    maintainer="Dalya Baron",
    maintainer_email="dalyabaron@gmail.com",
    license="BSD",
    packages=setuptools.find_packages(),

    install_requires=[
        "numpy >= 1.13",
        "scikit-learn >= 0.20",
        "scipy >= 1.0",
        "networkx >= 2.4",
        "joblib >= 0.13.2",
    ],
    python_requires='>=3.6',
)