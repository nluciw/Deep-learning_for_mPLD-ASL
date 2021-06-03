import setuptools

setuptools.setup(
    name="CBFnet",
    version="0.0.1",
    author="Nicholas Luciw",
    author_email="nicholas.luciw@mail.utoronto.ca",
    description="A CNN-based tool for blood flow estimation from MRI",
    url="https://github.com/nluciw/Deep-learning_for_mPLD-ASL",
    long_description=open("README.md").read(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU  General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Unix Shell',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    install_requires=[
        'nibabel', 'nipype', 'argparse', 'argcomplete', 'nilearn'
    ],
    extras_require={
        "cbfnet": ["tensorflow==1.15"],
        "cbfnet_gpu": ["tensorflow-gpu==1.15"],
    },
)