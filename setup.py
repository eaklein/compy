import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="compy",
    version="1.0.1",
    author="E. A. Klein",
    author_email="eklein@mit.edu",
    description="A package for reading CoMPASS directories into Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eaklein/compy",
    project_urls={
        "Bug Tracker": "https://github.com/eaklein/compy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    entry_points = {
        'console_scripts': [
            'compy-run=compy.compy_script:main'
        ],
    },
    install_requires=[
        'xmltodict',
        'click',
        'numpy',
        'matplotlib',
	'numpy',
	'pandas'
    ],
    python_requires=">=3.6",
)
