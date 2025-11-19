import sys
import traceback
from typing import Any, Iterable

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)


class DeviceOpTrace(TorchDispatchMode):
    """
    A context that prints a Python stack trace whenever a PyTorch op
    is dispatched on a particular device (default: ``"cpu"``).

    You can optionally filter which ops are logged, and toggle the
    tracer on or off with a flag.  Op names can be specified in several
    forms, all of which are matched:

    - Full overload name: ``\"aten.randn.default\"`` (``str(func)``).
    - Qualified name: ``\"aten::randn\"``.
    - Overload name: ``\"randn.default\"``.
    - Bare op name: ``\"randn\"``.

    - ``include_ops``: if set, only ops whose name matches at least one
      of the given strings are logged.
    - ``exclude_ops``: if set, ops whose name matches any of the given
      strings are never logged.

    Example
    -------
    >>> import torch
    >>> from pt_cpu import DeviceOpTrace
    >>> x = torch.randn(3, 3, device="cuda")
    >>> with DeviceOpTrace("cpu", include_ops={"aten.randn.default"}):
    ...     y = torch.randn(2, 2)  # will print a stack trace
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        print_once: bool = False,
        enabled: bool = True,
        include_ops: Iterable[str] | None = None,
        exclude_ops: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.print_once = print_once
        self.enabled = enabled
        self.include_ops = set(include_ops) if include_ops is not None else None
        self.exclude_ops = set(exclude_ops) if exclude_ops is not None else None
        self._has_printed = False
        self.printed_lines = set()

    def __enter__(self):
        if not self.enabled:
            # Do not activate TorchDispatchMode at all.
            return self
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            # We never entered the underlying mode stack.
            return False
        return super().__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _op_name_variants(func: Any) -> set[str]:
        names: set[str] = set()

        # Primary string form, e.g. "aten.randn.default"
        names.add(str(func))

        for attr in ("__name__", "__qualname__"):
            if hasattr(func, attr):
                value = getattr(func, attr)
                if isinstance(value, str):
                    names.add(value)

        overloadpacket = getattr(func, "overloadpacket", None)
        if overloadpacket is not None:
            names.add(str(overloadpacket))
            qn = getattr(overloadpacket, "__qualname__", None)
            if isinstance(qn, str):
                names.add(qn)

        # Derive simpler forms: strip namespace and overload where possible.
        derived = set(names)
        for name in names:
            # Drop "aten::" namespace if present.
            if "::" in name:
                derived.add(name.split("::", 1)[1])

            # Drop overload suffix if present.
            if "." in name:
                head, _ = name.split(".", 1)
                derived.add(head)

        return derived

    def _should_log_op(self, func: Any) -> bool:
        op_names = self._op_name_variants(func)

        if self.include_ops is not None and not (op_names & self.include_ops):
            return False

        if self.exclude_ops is not None and (op_names & self.exclude_ops):
            return False

        return True

    def _uses_tracked_device(
        self, args: Any, kwargs: dict[str, Any] | None, result: Any
    ) -> bool:
        if kwargs is None:
            kwargs = {}

        if all(t.device == self.device for t in _iter_tensors(kwargs)):
            return True

        device_kw = kwargs.get("device")
        if device_kw is not None and torch.device(device_kw) == self.device:
            return True

        return False

    def print_stack_without_last(self):
        # Extract the stack frames
        stack = traceback.extract_stack()

        # Remove the last frame (which corresponds to the call to print_stack_without_last itself)
        if stack:
            stack = stack[:-2]

        # Format and print the modified stack
        formatted_stack = "".join(traceback.format_list(stack))
        sys.stderr.write(formatted_stack)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = func(*args, **kwargs)

        if self._uses_tracked_device(args, kwargs, result) and self._should_log_op(func):
            stack = traceback.extract_stack()
            if len(stack) >= 2:
                # Remove the last two frames (which correspond to the call to __torch_dispatch__ itself)
                stack_line = stack[-2]
                stack_line_str = str(stack_line)
                if not stack_line_str in self.printed_lines:
                    self.printed_lines.add(stack_line_str)
                    if not self.print_once or not self._has_printed:
                        self._has_printed = True
                        print("===================================================")
                        print(f"[DeviceOpTrace] PyTorch op on {self.device}: {func}")
                        print("---------------------------------------------------")
                        self.print_stack_without_last()
                        print("===================================================")

        return result
