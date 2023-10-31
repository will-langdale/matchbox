from setuptools import find_packages, setup

setup(
    name="company-matching-framework",
    packages=find_packages(),
    version="0.1.0",
    description="A framework for orchestrating and comparing various"
    " company matching methodologies.",
    author="DDaT Data Science @ DBT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
