from os.path import realpath, dirname, join as path_join
from setuptools import setup, find_packages


name = "ARSpy"
description = "ARSpy"
long_description = "Adaptive Rejection Sampling for Python - Matlab Style"
maintainer = "Moritz Freidank"
maintainer_email = "freidankm@googlemail.com"

url = "https://github.com/MFreidank/ARSpy"

version = "0.1"

project_root = dirname(realpath(__file__))
requirements_file = path_join(project_root, "requirements.txt")

with open(requirements_file, "r") as f:
    install_requirements = f.read().splitlines()

setup_requirements = ["pytest-runner"]
test_requirements = ["pytest", "pytest-cov"]


if __name__ == "__main__":
    setup(
        name=name,
        version=version,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        description=description,
        url=url,
        download_url="https://github.com/MFreidank/ARSpy/archive/0.1.tar.gz",
        long_description=long_description,
        packages=find_packages(),
        keyword=["sampling", "adaptive rejection sampling", "adaptive", "rejection", "ars"],
        # package_data={"docs": ["*"]},
        # include_package_data=True,
        install_requires=install_requirements,
        setup_requires=setup_requirements,
        tests_require=test_requirements,
    )
