import pkgutil

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    # _module = loader.find_module(module_name).load_module(module_name)
    # globals()[module_name] = _module

from . import (
    metrics,
    utils_general,
    utils_experiment_parameters,
)

from . import (
    utils_prepare_data,
)
from . import models

from . import (
    run,
    experiment_routine,
)

from .graphic import (
    utils_results_data,
    style_utility,

)
from .graphic import (
    graphic_utility,

)
from .graphic import (
    graphic_utility,
)

__version__ = '0.1.0'
