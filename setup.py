from setuptools import setup

setup(name='counterfactualEO',
      version='0.1',
      description='Post-process binary predictors to satisfy counterfactual'
                  'equalized odds.',
      url='',
      author='Alan Mishler',
      author_email='alan@alanmishler.com',
      license='MIT',
      packages=['counterfactualEO'],
      install_requires=['cvxpy',
                        'itertools',
                        'numpy',
                        'pd',
                        'pickle',
                        'scipy',
                        'seaborn',
                        'sklearn',
                        'unittest'],
      zip_safe=False)