from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="learning-qp",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=read_requirements('requirements.txt'),
)
