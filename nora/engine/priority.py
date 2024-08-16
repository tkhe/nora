# Copyright (c) OpenMMLab. All rights reserved.

from enum import IntEnum
from typing import Union

__all__ = ["Priority"]


class Priority(IntEnum):
    """
    Hook priority levels.
    """

    VERY_HIGH = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    VERY_LOW = 100

    @staticmethod
    def get_priority(priority: Union[int, str, "Priority"]) -> int:
        """
        Get priority value.

        Args:
            priority (int or str or :obj:`Priority`): Priority.

        Returns:
            int: The priority value.
        """
        if isinstance(priority, int):
            if priority < 0 or priority > 100:
                raise ValueError(f"priority must be between 0 and 100, got {priority}!")
            return priority
        elif isinstance(priority, str):
            return Priority[priority.upper()].value
        elif isinstance(priority, Priority):
            return priority
        else:
            raise ValueError("`priority` must be an integer, string or Priority enum value")
