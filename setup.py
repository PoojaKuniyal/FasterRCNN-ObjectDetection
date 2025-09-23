from setuptools import setup, find_packages
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="gun_object_detection",
    version="0.1",
    author='Pooja',
    packages=find_packages(),
    install_requires = requirements
)