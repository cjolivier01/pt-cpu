import torch

from pt_cpu import DeviceOpTrace


def main() -> None:
    print("Running example with DeviceOpTrace on CPU")
    with DeviceOpTrace("cpu"):
        # This will allocate on CPU and trigger a stack trace.
        x = torch.randn(4, 4)
        y = x.relu()
        print("Computation result (CPU):", y.sum())

    print("Outside of DeviceOpTrace context; no tracing now.")
    z = torch.randn(2, 2)
    print("Result without tracing:", z.sum())


if __name__ == "__main__":
    main()

