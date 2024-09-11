from setuptools import Extension, setup
import numpy as np

# Definition of extension modules

DecisionTreeC = Extension('DecisionTreeC',
                sources = [
                    'src/decision_tree_c/decision_tree.cpp',
                    'src/decision_tree_c/decision_tree_wrapper.cpp',
                    'src/decision_tree_c/csr_matrix.cpp'
                ],
                include_dirs = ['include', np.get_include()],
                extra_compile_args=[
                    "--std=c++17",
                    "-Wno-sign-compare",
                    "-Wno-writable-strings",
                    "-DDEBUG", # Comment this one to remove debug logs
                ])

setup(ext_modules = [DecisionTreeC])
