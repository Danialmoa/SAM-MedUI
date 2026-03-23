from setuptools import setup, find_packages


def parse_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="SAMUI",
    version="0.1",
    description="Interactive GUI for myocardial scar segmentation using SAM on LGE-CMR images",
    author="Danial Moafi, Aida Moafi",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=parse_requirements(),
)
