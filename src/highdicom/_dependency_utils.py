from importlib import import_module, metadata
from packaging.requirements import Requirement
from types import ModuleType


def import_optional_dependency(
    module_name: str,
    feature: str
) -> ModuleType:
    """Import an optional dependency.

    This function is designed to support interaction with other common
    libraries that are not required for `highdicom` by default.

    Parameters
    ----------
    module_name: str
        Name of the module to be imported.
    feature: str
        Name or description of the feature that requires this dependency.
        This is used for improving the clarity of error messages.

    Returns
    -------
    ModuleType:
        Imported module.

    Raises
    ------
    ImportError:
        When the specified module cannot be imported.

    """
    for req_str in metadata.requires('highdicom'):
        req = Requirement(req_str)
        if req.name == module_name:
            break

    else:
        raise ValueError(
            f'`{module_name}` is not a requirement of highdicom '
            f'but is required for {feature}.'
        )

    try:
        module = import_module(name=module_name)

    except ImportError as error:
        raise ImportError(
            f'Optional dependency `{module_name}` could not be imported'
            f' but is required for {feature}.'
            f' highdicom requires {module_name}{req.specifier}.'
        ) from error

    installed_version = metadata.version(module_name)

    if installed_version not in req.specifier:
        raise ImportError(
            f'Optional dependency `{module_name}` has an unsuitable '
            f'version. Found {module_name}=={installed_version}, but '
            f'highdicom requires {module_name}{req.specifier}.'
        )

    return module
