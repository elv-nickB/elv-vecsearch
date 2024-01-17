from setuptools import setup, find_packages

setup(
    name="elv-vecsearch",
    version="0.1",
    packages=find_packages(), 
    install_requires=['schema==0.7.5', 
                      'faiss-gpu==1.7.2',
                      'Flask==3.0.0',
                      'huggingface-hub==0.17.3',
                      'sentence-transformers==2.2.2',
                      'torch==1.11.0',
                      'requests==2.31.0',
                      'marshmallow==3.20.2'
                      'scikit-learn==1.3.2',
                      'dill==0.3.7',
                      'elv-client-py @ git+https://github.com/eluv-io/elv-client-py.git@nick#egg=elv-client-py',
                      ]
)