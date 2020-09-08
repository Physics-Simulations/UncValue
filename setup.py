import setuptools
from uncvalue import __version__

# https://packaging.python.org/tutorials/packaging-projects/

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uncvalue",
    version=__version__,
    author="Physics Simulations",
    author_email="apuntsdefisica@gmail.com",
    description="Simple python class to propagate uncertainty in calculations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/labay11/Uncertainty-Value-Calculator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    install_requires=[
        'numpy>=1.18'
    ],
    python_requires='>=3.6',
)
