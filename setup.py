import setuptools

with open("README.txt", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nuBall",
    version="0.1.0",
    author="Krzysztof Miernik",
    author_email="k.a.miernik@gmail.com",
    description="nuBall experiment tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="",
    packages=setuptools.find_packages(),
    package_data={
                    'PicoNuclear': ['docs/*.xml']
                    },
    scripts=['bin/nucubes.py', 'bin/qtnuSpectra.py'],
    #entry_points={'console_scripts': ['nucubes = nuBall.nucubes'] },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.6',
)
