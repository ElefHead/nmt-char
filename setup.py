"""
Setup
"""
from setuptools import setup, find_packages

name = "nmt_char"
description = "Neural Machine Translation with Character level decoder"

version = "0.0.1"
requirements = []

with open('requirements.txt', 'r') as req:
    requirements = [line for line in req if line[0].isalpha()]

setup(
    name=name,
    version=version,
    author="Ganesh Jagadeesan",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    description=description
)
