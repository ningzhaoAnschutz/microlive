"""Pure, Qt-free frame-range model for the Import-tab crop feature.

This implements the "crop is the movie" model. A :class:`FrameRange` describes an
inclusive ``[start, end]`` selection over a loaded acquisition of
``source_total_frames`` frames. When a crop is applied, the selected region becomes
the active movie and active frame ``0`` corresponds to source frame ``start``.

The module is intentionally free of any GUI/Qt/NumPy-array dependencies so the range
and validation logic can be unit-tested without importing the application.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass


def _as_index(value, name):
    """Return ``value`` as a plain ``int`` if it is integral, else raise.

    Accepts Python ``int`` and NumPy integer types, and integral floats
    (e.g. ``3.0``). Rejects booleans, non-integral floats, and other types
    rather than silently repairing them.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if as_float.is_integer():
            return int(as_float)
        raise TypeError(f"{name} must be integral, got {value!r}")
    raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


@dataclass(frozen=True)
class FrameRange:
    """An inclusive, validated frame selection over a source acquisition.

    Attributes:
        source_total_frames: Number of frames in the loaded acquisition (>= 1).
        start: Inclusive source start index (0-based).
        end: Inclusive source end index (0-based).
    """

    source_total_frames: int
    start: int
    end: int

    def __post_init__(self):
        object.__setattr__(self, "source_total_frames", _as_index(self.source_total_frames, "source_total_frames"))
        object.__setattr__(self, "start", _as_index(self.start, "start"))
        object.__setattr__(self, "end", _as_index(self.end, "end"))
        if self.source_total_frames < 1:
            raise ValueError(f"source_total_frames must be >= 1, got {self.source_total_frames}")
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end > self.source_total_frames - 1:
            raise ValueError(
                f"end ({self.end}) must be <= source_total_frames - 1 ({self.source_total_frames - 1})"
            )
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")

    # ---- derived quantities -------------------------------------------------
    @property
    def stop(self) -> int:
        """Python-slice endpoint (exclusive): ``end + 1``."""
        return self.end + 1

    @property
    def count(self) -> int:
        """Number of frames in the active (cropped) movie."""
        return self.end - self.start + 1

    @property
    def is_full(self) -> bool:
        """True when the selection spans the whole source movie."""
        return self.start == 0 and self.end == self.source_total_frames - 1

    # ---- factories / helpers ------------------------------------------------
    @classmethod
    def full_range(cls, source_total_frames) -> "FrameRange":
        """Return the full, uncropped range for a movie of ``source_total_frames``."""
        n = _as_index(source_total_frames, "source_total_frames")
        return cls(n, 0, n - 1)

    def slice_for_source_stack(self) -> slice:
        """Return ``slice(start, end + 1)`` for slicing a TZYXC source stack's time axis."""
        return slice(self.start, self.stop)

    def signature(self) -> tuple:
        """Return a hashable, serializable identity for this range."""
        return (self.source_total_frames, self.start, self.end)


def is_full_selection(frame_range) -> bool:
    """Return True when there is no meaningful crop.

    A ``None`` range (no image loaded) is treated as *not cropped*.
    """
    if frame_range is None:
        return False
    return frame_range.is_full
