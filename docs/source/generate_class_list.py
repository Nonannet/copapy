# This script generates the source md-files for all classes and functions for the docs

import importlib
import inspect
import fnmatch
from io import TextIOWrapper
import os
import argparse


def write_manual(f: TextIOWrapper, doc_files: list[str], title: str) -> None:
    write_dochtree(f, title, doc_files)


def write_classes(f: TextIOWrapper, patterns: list[str], module_name: str, title: str, description: str = '', exclude: list[str] = [], api_dir: str = 'api') -> None:
    """Write the classes to the file."""
    module = importlib.import_module(module_name)

    classes = [
        name for name, obj in inspect.getmembers(module, inspect.isclass)
        if (any(fnmatch.fnmatch(name, pat) for pat in patterns if name not in exclude) and
            obj.__doc__ and '(Automatic generated stub)' not in obj.__doc__)
    ]

    if description:
        f.write(f'{description}\n\n')

    write_dochtree(f, title, classes)

    for cls in classes:
        with open(f'{api_dir}/{cls}.md', 'w') as f2:
            f2.write(f'# {module_name}.{cls}\n')
            f2.write('```{eval-rst}\n')
            f2.write(f'.. autoclass:: {module_name}.{cls}\n')
            f2.write('   :members:\n')
            f2.write('   :undoc-members:\n')
            f2.write('   :show-inheritance:\n')
            f2.write('   :inherited-members:\n')
            f2.write('```\n\n')


def write_functions(f: TextIOWrapper, patterns: list[str], module_name: str, title: str, description: str = '', exclude: list[str] = [], path_patterns: list[str] = ['*'], api_dir: str = 'api') -> None:
    """Write the classes to the file."""
    module = importlib.import_module(module_name)

    functions: list[str] = []
    for name, fu in inspect.getmembers(module, inspect.isfunction):
        if (any(fnmatch.fnmatch(name, pat) for pat in patterns if name not in exclude)):
            path = inspect.getfile(fu)
            if any(fnmatch.fnmatch(path, pat) for pat in path_patterns):
                functions.append(name)


    if description:
        f.write(f'{description}\n\n')

    write_dochtree(f, title, functions)

    for func in functions:
        if not func.startswith('_'):
            with open(f'{api_dir}/{func}.md', 'w') as f2:
                f2.write(f'# {module_name}.{func}\n')
                f2.write('```{eval-rst}\n')
                f2.write(f'.. autofunction:: {module_name}.{func}\n')
                f2.write('```\n\n')


def write_dochtree(f: TextIOWrapper, title: str, items: list[str]):
    f.write('```{toctree}\n')
    f.write(':maxdepth: 1\n')
    f.write(f':caption: {title}:\n')
    #f.write(':hidden:\n')
    for text in items:
        if not text.startswith('_'):
            f.write(f"{text}\n")
    f.write('```\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate class and function documentation')
    parser.add_argument('--api-dir', type=str, default='docs/source/api', help='Output directory for API documentation (default: api)')
    args = parser.parse_args()

    api_dir = args.api_dir

    # Ensure the output directory exists
    os.makedirs(api_dir, exist_ok=True)

    with open(f'{api_dir}/index.md', 'w') as f:
        f.write('# User API\n\n')

        write_classes(f, ['*'], 'copapy', title='Classes', api_dir=api_dir)

        write_functions(f, ['*'], 'copapy', title='General functions', path_patterns=['*_autograd.py', '*_basic_types.py', '*_target.py'], api_dir=api_dir)

        write_functions(f, ['*'], 'copapy', title='Math functions', path_patterns=['*_math*'], exclude=['get_42'], api_dir=api_dir)

        write_functions(f, ['*'], 'copapy', title='Vector functions', path_patterns=['*_vectors*'], api_dir=api_dir)

        write_functions(f, ['*'], 'copapy', title='Tensor/Matrix functions', path_patterns=['*_tensors*'], api_dir=api_dir)

        #write_manual(f, ['NumLike'], title='Types')

    with open(f'{api_dir}/backend.md', 'w') as f:
        f.write('# Backend\n\n')
        write_classes(f, ['*'], 'copapy.backend', title='Classes', api_dir=api_dir)

        write_functions(f, ['*'], 'copapy.backend', title='Functions', api_dir=api_dir)
