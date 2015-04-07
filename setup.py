from setuptools import setup
import os

#-Write Versions File-#
#~~~~~~~~~~~~~~~~~~~~~#

VERSION = '0.2.1-alpha'

def write_version_py(filename=None):
    """This constructs a version file for the project"""
    doc = "\"\"\"This is a VERSION file and should NOT be manually altered\"\"\""
    doc += "\nversion = '%s'" % VERSION
    
    if not filename:
        filename = os.path.join(os.path.dirname(__file__), 'pycollocation', 'version.py')
    
    fl = open(filename, 'w')
    try:
        fl.write(doc)
    finally:
        fl.close()

write_version_py()  # This is a file used to control the pycollocation.__version__ attribute

setup(
    name="pyCollocation",
    version=VERSION,
    license="MIT License",
    author="davidrpugh",
    install_requires=["numpy",
                      "scipy",
                      "sympy",
                      "pandas",
                      "ipython",
                     ],
    author_email="david.pugh@maths.ox.ac.uk",
    classifiers=['Development Status :: 1 - Planning',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 ]
    )
