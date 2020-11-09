import setuptools

setuptools.setup(
    name="src",
    version="0.1.1",
    author="Moritz Wagner",
    author_email="moritzwagner95@hotmail.de",
    description="package for deep survival",
    url="https://github.com/MoritzWag/DeepSurvival",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    zip_safe=False
)