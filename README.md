# pt-cpu

`pt-cpu` is a tiny helper library that lets you debug where PyTorch operations are executed by printing a Python stack trace whenever an op touches a given device (e.g. the CPU).

It is built on top of `TorchDispatchMode`, so it works for both factory functions and regular tensor ops without requiring any tensor subclassing.

## Installation

After this repository is pushed to GitHub, you can install it into an environment that already has PyTorch:

```bash
pip install git+ssh://git@github.com/cjolivier01/pt-cpu.git
```

Or from a local clone:

```bash
git clone ssh://git@github.com/cjolivier01/pt-cpu.git
cd pt-cpu
pip install -e .
```

## Basic usage

The main entry point is the `DeviceOpTrace` context manager:

```python
import torch
from pt_cpu import DeviceOpTrace

x = torch.randn(3, 3, device="cuda")

with DeviceOpTrace("cpu"):
    # Any op that uses CPU tensors or a device="cpu" kwarg
    # will print a Python stack trace to stderr.
    y = torch.randn(4, 4)  # allocates on CPU
    z = y.relu()
```

When run, you will see lines similar to:

```text
[DeviceOpTrace] PyTorch op on cpu: aten.randn.default
<full Python stack trace>
```

## Filtering by ops

You can restrict which ops are logged using `include_ops` and `exclude_ops`. Op names can be specified in several forms:

- `"aten.randn.default"` (full overload name, `str(func)`)
- `"aten::randn"` (qualified name)
- `"randn.default"` (overload name)
- `"randn"` (bare op name)

```python
import torch
from pt_cpu import DeviceOpTrace

# Only log randn on CPU (bare name works)
with DeviceOpTrace("cpu", include_ops={"randn"}):
    a = torch.randn(2, 2)  # logged
    b = a.relu()           # not logged

# Log everything on CPU except relu
with DeviceOpTrace("cpu", exclude_ops={"relu"}):
    c = torch.randn(2, 2)  # logged
    d = c.relu()           # not logged
```

## Log only the first occurrence

If you only care about the first time a CPU op is hit, set `print_once=True`:

```python
with DeviceOpTrace("cpu", print_once=True):
    x = torch.randn(2, 2)   # traced
    y = x + 1               # NOT traced (already printed once)
```

## Example script

The repository includes a small script in `example.py` that demonstrates basic usage:

```bash
python example.py
```

This will run a short CPU computation inside `DeviceOpTrace("cpu")` and print the corresponding stack traces. 

