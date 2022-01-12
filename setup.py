import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dnmf-python",
    version="0.0.3",
    author="Andrej Janchevski",
    author_email="andrej.janchevski@epfl.ch",
    description="Unofficial Python implementation of the DNMF overlapping community detection algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bani57/dnmf-python",
    project_urls={
        "Bug Tracker": "https://github.com/Bani57/dnmf-python/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.1",
    ],
)
