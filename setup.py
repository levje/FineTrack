from setuptools import setup
from torch.utils import cpp_extension

setup(
    packages=['finetrack'],

    # List C++ extensions
    ext_modules=[
        cpp_extension.CppExtension(
            'finetrack.algorithms.shared.disc_cumsum',
            ['finetrack/algorithms/shared/disc_cumsum.cpp'],
        ),
    ],

    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
