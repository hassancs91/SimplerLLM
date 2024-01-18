from setuptools import setup, find_packages

setup(
    name='SimplerLLM',  # Replace with your library's name
    version='0.1.1',  # Your library's initial version
    author='LearnWithHasan',  # Your name or your organization's name
    author_email='hasan@learnwithhasan.com',  # Your contact email
    description='An asynchronous text generation library using OpenAI.',  # A short description
    long_description=open('README.md').read(),  # Long description read from the the readme file
    long_description_content_type='text/markdown',  # Type of the long description
    url='https://github.com/yourusername/text_generator',  # Link to your project's repository
    packages=find_packages(),  # Automatically find your package
    # install_requires=[
    #     'openai>=0.2.4',  # Ensure you pin down the versions for better reproducibility
    #     'python-dotenv>=0.15.0',
    #     'asyncio',
    # ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Choose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',  # Your license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions you support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',  # Minimum version requirement of the package
    keywords='text generation, openai, asyncio',  # Short descriptions of your library
    project_urls={  # Additional URL for the project
        'Source': 'https://github.com/yourusername/text_generator',
    },
)
