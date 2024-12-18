from setuptools import setup
from torch.utils import cpp_extension

setup(
    packages=['FineTrack'],

    # List C++ extensions
    ext_modules=[
        cpp_extension.CppExtension(
            'FineTrack.algorithms.shared.disc_cumsum',
            ['FineTrack/algorithms/shared/disc_cumsum.cpp'],
        ),
    ],

    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
