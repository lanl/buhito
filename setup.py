import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

lib_path = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(lib_path, 'requirements.txt')
with open(requirements_path) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="buhito",
    version="0.0.0",
    author="Buhito Developers",
    author_email="un_graphlets@lanl.gov",
    description="Graphlet enumeration and analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lanl/buhito",
    project_urls={
        "Bug Tracker": "https://github.com/lanl/buhito/issues",
        "Source Code": "https://github.com/lanl/buhito",
    },
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
)