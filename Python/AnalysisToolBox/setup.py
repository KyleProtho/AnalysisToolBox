# Description: Setup file for AnalysisToolbox package.
from setuptools import setup, find_packages

# Read the README file.
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='AnalysisToolbox',
    description='A collection tools in Python for data collection and processing, statisitics, analytics, and intelligence analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/KyleProtho/AnalysisToolBox/tree/master/Python/AnalysisToolBox",
    version='0.1.0',
    author='Kyle Protho',
    author_email='kyletprotho@gmail.com',
    license='GNU GPLv3',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Analysts',
        'Topic :: Intelligence Analysis, Data Science, Statistics',
        'License :: OSI Approved :: GNU GPLv3 License',
        'Programming Language :: Python :: >=3.9',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.9',
    packages=find_packages(where="AnalysisToolbox"),
    install_requires=[
        'beautifulsoup4',
        'dotenv',
        'folium',
        'fuzzywuzzy',
        'geopandas',
        'io',
        'Jinja2',
        'json',
        'Levenshtein',
        'lida',
        'lifelines',
        'mapclassify',
        'math',
        'matplotlib',
        'mlxtend',
        'numpy',
        'openai',
        'os',
        'pandas',
        'pygris',
        'PyPDF2',
        'pywin32',
        're',
        'requests',
        'scipy',
        'seaborn',
        'scikit-learn',  # sklearn
        'statsmodels',
        'sympy',
        'tableone',
        'textwrap',
        'zipfile',
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
            'twine>=4.0.2'
        ],
    },
)
