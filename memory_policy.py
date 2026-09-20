"""Scoped memory reservations; importing the package never changes ComfyUI."""

from contextvars import ContextVar
from functools import wraps
import inspect
import logging
import math
import threading

logger = logging.getLogger(__name__)
_active_reservation = ContextVar("myang_memory_reservation", default=None)
_reservation_lock = threading.RLock()


def _gigabytes(value):
    value = float(value or 0)
    if not math.isfinite(value) or value < 0:
        raise ValueError("VRAM reservation must be a finite non-negative number")
    return value


class MemoryReservation:
    """Restore the actual previous target, including on cancellation or OOM.

    Older AIMDO libraries expose a setter without a getter. Only use that
    setter when ComfyUI supplied an explicit --reserve-vram baseline; otherwise
    retain the backend policy and let per-model page eviction handle pressure.
    Never change the NVML pressure source or reinitialize the AIMDO library.
    """

    def __init__(self, reserve_gb=0):
        self.reserve_gb = _gigabytes(reserve_gb)
        self.setter = None
        self.previous_headroom = None
        self.changed = False
        self.current_headroom = None

    def __enter__(self):
        import comfy.memory_management as memory
        import comfy.model_management as management
        _reservation_lock.acquire()
        self.management = management
        self.previous_reserved = management.EXTRA_RESERVED_VRAM
        self.dynamic = bool(getattr(memory, "aimdo_enabled", False))
        parent = _active_reservation.get()
        self.token = _active_reservation.set(self)
        try:
            if self.dynamic:
                try:
                    import comfy_aimdo.control as aimdo
                    lib = getattr(aimdo, "lib", None)
                    setter = getattr(lib, "set_simple_vram_headroom", None)
                    getter = getattr(lib, "get_simple_vram_headroom", None)
                    if callable(setter):
                        if parent is not None and parent.current_headroom is not None:
                            self.previous_headroom = parent.current_headroom
                        elif callable(getter):
                            self.previous_headroom = int(getter())
                        else:
                            from comfy.cli_args import args
                            baseline = getattr(args, "reserve_vram", None)
                            if baseline is not None:
                                self.previous_headroom = int(_gigabytes(baseline) * 1024 ** 3)
                        if self.previous_headroom is not None:
                            self.setter = setter
                            self.current_headroom = self.previous_headroom
                except Exception as error:
                    logger.debug("AIMDO reservation unavailable: %s", error)
            self.increase(self.reserve_gb)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def increase(self, reserve_gb):
        requested = int(_gigabytes(reserve_gb) * 1024 ** 3)
        if not self.dynamic:
            self.management.EXTRA_RESERVED_VRAM = max(
                self.management.EXTRA_RESERVED_VRAM, self.previous_reserved, requested)
            return True
        if self.setter is None or requested <= 0:
            return False
        # Record the attempt before calling a foreign function: a failing
        # setter may have partially applied its change.
        self.changed = True
        try:
            self.current_headroom = max(self.current_headroom, requested)
            self.setter(self.current_headroom)
            return True
        except Exception as error:
            logger.warning("Could not apply AIMDO headroom: %s", error)
            return False

    def __exit__(self, *exc):
        try:
            self.management.EXTRA_RESERVED_VRAM = self.previous_reserved
            if self.changed:
                try:
                    self.setter(self.previous_headroom)
                except Exception as error:
                    logger.warning("Could not restore AIMDO headroom: %s", error)
        finally:
            _active_reservation.reset(self.token)
            _reservation_lock.release()


def increase_reservation(reserve_gb):
    scope = _active_reservation.get()
    return scope.increase(reserve_gb) if scope is not None else False


def scoped_reservation(function):
    """Scope all work, including noise generation and VAE preparation."""
    signature = inspect.signature(function)

    @wraps(function)
    def run(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        reserve = arguments.arguments.get("reserve_vram_gb", 0)
        label = arguments.arguments.get("pass_label")
        if label is not None and not str(label).startswith("sample2"):
            import comfy.memory_management as memory
            if getattr(memory, "aimdo_enabled", False):
                reserve = 0
        with MemoryReservation(reserve):
            return function(*args, **kwargs)
    return run
