from typing import Optional

class SICUError(Exception):
    """Base class for errors raised by SICU."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)

class UnitaryError(SICUError):
    """Error raised when a unitary matrix is expected but not found."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)