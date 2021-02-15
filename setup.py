from setuptools import setup
exec(open('rlxfinance/version.py').read())

setup(name='rlxfinance',
      version=__version__,
      description='rlx finance tools',
      url='http://github.com/rramosp/rlxfinance',
      install_requires=['matplotlib','numpy', 'pandas','joblib',
                        'progressbar2', 'psutil', 'bokeh', 'pyshp',
                        'statsmodels', 'filterpy'],
      scripts=[],
      author='rlx',
      author_email='rulix.rp@gmail.com',
      license='MIT',
      packages=['rlxfinance'],
      include_package_data=True,
      zip_safe=False)
