import os
import re

from setuptools import setup, find_packages


def read(*names, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *names)) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('requirements.txt') as f:
    required = f.read().splitlines()

VERSION = find_version('src', 'fairnesseval', '__init__.py')
long_description = read('README.md')

setup(
    name='fairnesseval',
    version=VERSION,
    description='This repository is dedicated to the evaluation and testing of a novel fairness approach in machine learning. Automated decision-making systems can potentially introduce biases, raising ethical concerns. This has led to the development of numerous bias mitigation techniques. However, the selection of a fairness-aware model for a specific dataset often involves a process of trial and error, as it is not always feasible to predict in advance whether the mitigation measures provided by the model will meet the user\'s requirements, or what impact these measures will have on other model metrics such as accuracy and run time. Existing fairness toolkits lack a comprehensive benchmarking framework. To bridge this gap, we present FairnessEval, a framework specifically designed to evaluate fairness in Machine Learning models. FairnessEval streamlines dataset preparation, fairness evaluation, and result presentation, while also offering customization options. In this demonstration, we highlight the functionality of FairnessEval in the selection and validation of fairness-aware models. We compare various approaches and simulate deployment scenarios to showcase FairnessEval effectiveness.',
    long_description=long_description,
    author='Andrea Baraldi, Matteo Brucato, Miroslav Dudík, Francesco Guerra, Matteo Interlandi',
    author_email='baraldian@gmail.com, mbrucato@microsoft.com, mdudik@microsoft.com, francesco.guerra@unimore.it, mainterl@microsoft.com',
    url='https://github.com/softlab-unimore/fairnesseval.git',
    packages=find_packages(where='src', exclude=('*test*', '*run_experiments*')),
    license='MIT',
    package_dir={'': 'src'},
    install_requires=required,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
)

