from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ML_Project",
    version="0.1.0",
    author="Siddhartha",
    packages=find_packages(),
    install_requires=requirements,
    description="A machine learning project setup",
    

)