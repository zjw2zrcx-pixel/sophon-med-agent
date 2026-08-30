from pathlib import Path
from setuptools import Extension, setup
import sysconfig

ROOT = Path(__file__).parent
TORCH = Path('/data/env310/lib/python3.10/site-packages/torch/include')

setup(
    name='sail-keyword-search',
    ext_modules=[Extension(
        'sail_keyword_search', [str(ROOT / 'sail_keyword_search.cc')],
        include_dirs=[str(TORCH), sysconfig.get_config_var('INCLUDEPY')],
        language='c++', extra_compile_args=['-std=c++17', '-O3'],
    )],
)
