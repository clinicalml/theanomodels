# incorporated from https://github.com/pypa/sampleproject/blob/master/setup.py

from setuptools import find_packages, setup

setup(
        name='theanomodels',
        version='1.0.0',
        description='A lightweight wrapper around theano for rapid-prototyping of machine learning models.',
        long_description=open('README.md').read(),
        url='https://github.com/clinicalml/theanomodels/tree/change_to_relative_imports',
        author='Rahul G. Krishnan, Justin Mao-Jones',
        author_email='rahul@cs.nyu.edu, justinmaojones@gmail.com',
        license='MIT',
        classifiers=[
            # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',
            'License :: OSI Approved :: MIT License',

            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
        ],
        keywords='sample setuptools development',
        packages=find_packages(exclude=['contrib', 'docs', 'tests']),
        install_requires=['Theano>=0.8.2',
                          'h5py>=2.4.0b1',
                          'numpy>=1.10.1',
                          'scipy>=0.18.1',
        ],
)

