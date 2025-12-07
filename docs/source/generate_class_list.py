# This script generates the source md-files for all classes and functions for the docs

import importlib
import inspect
import fnmatch
from io import TextIOWrapper
import os


def write_manual(f: TextIOWrapper, doc_files: list[str], title: str) -> None:
    write_dochtree(f, title, doc_files)


def write_classes(f: TextIOWrapper, patterns: list[str], module_name: str, title: str, description: str = '', exclude: list[str] = []) -> None:
    """Write the classes to the file."""
    module = importlib.import_module(module_name)

    classes = [
        name for name, obj in inspect.getmembers(module, inspect.isclass)
        if (any(fnmatch.fnmatch(name, pat) for pat in patterns if pat not in exclude) and
            obj.__doc__ and '(Automatic generated stub)' not in obj.__doc__)
    ]

    if description:
        f.write(f'{description}\n\n')

    write_dochtree(f, title, classes)

    for cls in classes:
        with open(f'docs/source/api/{cls}.md', 'w') as f2:
            f2.write(f'# {module_name}.{cls}\n')
            f2.write('```{eval-rst}\n')
            f2.write(f'.. autoclass:: {module_name}.{cls}\n')
            f2.write('   :members:\n')
            f2.write('   :undoc-members:\n')
            f2.write('   :show-inheritance:\n')
            f2.write('   :inherited-members:\n')
            f2.write('```\n\n')


def write_functions(f: TextIOWrapper, patterns: list[str], module_name: str, title: str, description: str = '', exclude: list[str] = []) -> None:
    """Write the classes to the file."""
    module = importlib.import_module(module_name)

    functions = [
        name for name, _ in inspect.getmembers(module, inspect.isfunction)
        if (any(fnmatch.fnmatch(name, pat) for pat in patterns if pat not in exclude))
    ]

    if description:
        f.write(f'{description}\n\n')

    write_dochtree(f, title, functions)

    for func in functions:
        if not func.startswith('_'):
            with open(f'docs/source/api/{func}.md', 'w') as f2:
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
    # Ensure the output directory exists
    os.makedirs('docs/source/api', exist_ok=True)

    with open('docs/source/api/index.md', 'w') as f:
        f.write('# Classes and functions\n\n')

        write_classes(f, ['*'], 'copapy', title='Classes')

        write_functions(f, ['*'], 'copapy', title='Functions')

        #write_manual(f, ['../ndfloat', '../floatarray'], title='Types')
