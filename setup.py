from distutils.core import setup


setup(
    name='topnum',
    packages=[
        'topnum',
        'topnum.data',
        'topnum.scores',
        'topnum.search_methods',
        'topnum.search_methods.topic_bank',
        'topnum.search_methods.topic_bank.phi_initialization',
        'topnum.tests'
    ],
    version='0.3.0',
    license='MIT',
    description='A set of methods for finding an appropriate number of topics in a text collection',
    author='Machine Intelligence Laboratory',
    author_email='vasiliy.alekseyev@phystech.edu',
    url='https://github.com/machine-intelligence-laboratory/OptimalNumberOfTopics',
    keywords=[
        'topic modeling',
        'document clustering',
        'number of clusters',
        'ARTM',
        'regularization',
    ],
    install_requires=[
        'anchor-topic==0.1.2',
        'bigartm>=0.9.2',
        'dill==0.3.8',
        'lapsolver==1.1.0',
        'matplotlib==3.7.5',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'protobuf==3.20.3',  # TODO: BigARTM dependency
        'pytest==8.1.1',
        'scikit-learn==1.3.2',
        'scipy==1.10.1',
        'topicnet>=0.9.0',
        'tqdm==4.66.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
)
