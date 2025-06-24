# Description: Setup file for AnalysisToolbox package.
from setuptools import setup, find_packages

# Read the README file.
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Setup the package.
setup(
    name='analysistoolbox',
    description='A collection tools in Python for data collection and processing, statistics, analytics, and intelligence analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/KyleProtho/AnalysisToolBox/tree/master/analysistoolbox",
    version='2.2.4',
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
    dependencies=[
        'beautifulsoup4',
        'censusgeocode',
        'edgar_tool',
        'folium',
        'fuzzywuzzy',
        'geopandas',
        'Jinja2',
        'Levenshtein',
        'lanchain',
        'langchain_anthropic',
        'langchain_core',
        'lida',
        'lifelines',
        'mapclassify',
        'matplotlib<3.8',
        'mlxtend',
        'numpy',
        'openai',
        'pandas',
        'pinecone',
        'psmpy',
        'pygris',
        'pymetalog',
        'PyPDF2',
        'python-dotenv',
        'pywin32',
        'requests',
        'scikit-learn',  # sklearn
        'scipy',
        'seaborn',
        'sentence_transformers',
        'statsmodels',
        'sympy',
        'tableone',
        'tensorflow',
        'tqdm',
        'xgboost',
        'yellowbrick'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
            'twine>=4.0.2'
        ],
    },
)
