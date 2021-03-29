from setuptools import find_packages
from distutils.core import setup

setup(
    name="model_pricing",
    version="0.1",
    packages=find_packages(include=["model_pricing", "model_pricing.*"]),
    description="Code for model pricing study",
)
