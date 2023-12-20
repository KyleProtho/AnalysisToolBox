# Description: Setup file for AnalysisToolbox package.
from setuptools import setup, find_packages

# Read the README file.
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Setup the package.
setup(
    name='analysistoolbox',
    description='A collection tools in Python for data collection and processing, statisitics, analytics, and intelligence analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/KyleProtho/AnalysisToolBox/tree/master/analysistoolbox",
    version='1.3.1',
    author='Kyle Protho',
    author_email='kyletprotho@gmail.com',
    license='GNU GPLv3',
    classifiers=[
        # "Development Status :: 1 - Alpha",
        # "Intended Audience :: Analysts",
        # "Topic :: Intelligence Analysis, Data Science, Statistics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    # package_dir={'': 'analysistoolbox'},
    packages=find_packages(),
    install_requires=[
        'beautifulsoup4',
        'folium',
        'fuzzywuzzy',
        'geopandas',
        'Jinja2==3.1.2',
        'Levenshtein',
        'lida',
        'lifelines',
        'mapclassify',
        'matplotlib',
        'mlxtend',
        'numpy==1.24.5',
        'openai',
        'pandas==1.5.3',
        'pygris',
        'pymetalog',
        'PyPDF2',
        'python-dotenv',
        'pywin32',
        'requests',
        'scipy==1.11.1',
        'seaborn',
        'scikit-learn==1.1.2',  # sklearn
        'statsmodels==0.14.0',
        'sympy',
        'tableone',
        'tensorflow',
        'xgboost==2.0.0',
        'yellowbrick'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
            'twine>=4.0.2'
        ],
    },
)
