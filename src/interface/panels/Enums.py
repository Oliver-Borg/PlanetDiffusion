from enum import Enum


# Each panel should be responsible for it's own sketch

class BrushModeEnum(Enum):
    """
    Brush mode enum.
    """

    BRUSH = 0
    ERASE = 1
