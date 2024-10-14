# Copyright (c) Facebook, Inc. and its affiliates.

import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from tabulate import tabulate

__all__ = ["Registry"]


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    user's custom modules.

    Registered object could be built from registry.

    Examples:

        >>> # create a registry
        >>> BACKBONE_REGISTRY = Resgistry("backbone")
        >>> # to register a module, used as a decorator
        >>> @BACKBONE_REGISTRY.register()
        >>> class MyBackbone:
        >>>     pass
        >>> # to register a module, used as a function call
        >>> BACKBONE_REGISTRY.register("AnyBackbone", MyBackbone)
        >>> # build object from registry
        >>> obj = BACKBONE_REGISTRY.build(dict(_target_="AnyBackbone"))
    """

    def __init__(self, name: str, allow_override: bool = False):
        """
        Args:
            name (str): the name of registry.
            allow_override (bool): If True, the previously registered object will
                be overriden when the same name is registered.
        """
        self._name = name
        self._allow_override = allow_override

        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any):
        if name in self._obj_map and not self._allow_override:
            raise ValueError(f"An object named '{name}' was already registered in '{self._name}' registry!")

        self._obj_map[name] = obj

    def register(self, name: Optional[str] = None, obj: Optional[Any] = None):
        """
        Register the given object under the the name or `obj.__name__` if name is None.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                nonlocal name

                if name is None:
                    name = func_or_class.__name__

                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__

        self._do_register(name, obj)

    def keys(self) -> List[str]:
        return self._obj_map.keys()

    def values(self) -> List[str]:
        return self._obj_map.values()

    def items(self):
        return self._obj_map.items()

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)

        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")

        return ret

    def build(self, cfg, **extra_kwargs) -> Any:
        """
        Build an instance.

        Args:
            cfg (dict): config dict need to be built.

        Return:
            Any: The constructed object.
        """
        kwargs = copy.deepcopy(cfg)
        kwargs.update(**extra_kwargs)

        assert "_target_" in kwargs, "`cfg` or `extra_kwargs` must contain the key '_target_'"

        name = kwargs.pop("_target_")
        obj = self.get(name)

        try:
            return obj(**kwargs)
        except Exception as e:
            raise e

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __len__(self) -> int:
        return len(self._obj_map)

    def __repr__(self) -> str:
        table_headers = ["Name", "Objects"]
        table = tabulate(self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid")
        return f"Registry of {self._name}\n{table}"

    def __iter__(self):
        return iter(self._obj_map.items())

    def __str__(self) -> str:
        return self.__repr__()
