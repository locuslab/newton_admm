from setuptools import find_packages, setup

setup(
    name='newton_admm',
    version='0.0.1',
    description="Newton ADMM based cone solver",
    author='Eric Wong',
    author_email='ericwong@cs.cmu.edu',
    platforms=['any'],
    license="Apache 2.0",
    url='https://github.com/locuslub/newton_admm',
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
        'scipy',
        'block'
    ]
)
