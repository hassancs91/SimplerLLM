from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='SimplerLLM',  # Replace with your library's name
    version='0.1.0',  # Your library's initial version
    author='Hasan Aboul Hasan',  # Your name or your organization's name
    author_email='hasan@learnwithhasan.com',  # Your contact email
    description='An easy-to-use Library for interacting with language models.',  # A short description
    long_description=open('README.md').read(),  # Long description read from the the readme file
    long_description_content_type='text/markdown',  # Type of the long description
    url='https://github.com/hassancs91/SimplerLLM',  # Link to your project's repository
    packages=find_packages(),  # Automatically find your package
    install_requires=requirements,
    python_requires='>=3.6',  # Minimum version requirement of the package
    keywords='text generation, openai, LLM, RAG',  # Short descriptions of your library
)
