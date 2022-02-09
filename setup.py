import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="compy-EAKLEIN",
    version="0.0.1",
    author="E. A. Klein",
    author_email="eklein@mit.edu",
    description="A package for reading CoMPASS directories into Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eaklein/CoMPASS",
    project_urls={
        "Bug Tracker": "https://github.com/eaklein/CoMPASS/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)