from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op, Tensor, Parameter
from tvm import relax
from typing import Type, Union, Tuple


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: nn.Tensor):
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter((num_channels,))
        self.bias = nn.Parameter((num_channels,))
        self.eps = eps
    
    def forward(self, x: nn.Tensor):
        c = self.num_channels
        u = op.sum(x, axis=1, keepdims=True) / c
        diff = x - u
        var = op.sum(diff * diff, axis=1, keepdims=True) / c
        x_norm = diff / op.sqrt(var + self.eps)
        w = op.reshape(self.weight, (1, c, 1, 1))
        b = op.reshape(self.bias, (1, c, 1, 1))
        return x_norm * w + b


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter((in_channels, out_channels, kernel_size, kernel_size), "float32")
        self.bias = Parameter((out_channels,), "float32") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = relax.op.nn.conv2d_transpose(
            x._expr,
            self.weight._expr,
            strides=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            kernel_layout="IOHW",
            data_layout="NCHW",
        )

        if self.bias is not None:
            bias = relax.op.reshape(self.bias._expr, (1, self.out_channels, 1, 1))
            y = relax.op.add(y, bias)

        return op.wrap_nested(y, "conv2d_transpose")


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter((out_channels, in_channels, kernel_size[0], kernel_size[1]), "float32",)
        self.bias = Parameter((out_channels,), "float32") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = relax.op.nn.conv2d(
            x._expr,
            self.weight._expr,
            strides=self.stride,
            padding=self.padding,
            kernel_layout="OIHW",
            data_layout="NCHW",
            out_dtype="float32",
        )

        if self.bias is not None:
            bias = relax.op.reshape(self.bias._expr, (1, self.out_channels, 1, 1))
            y = relax.op.add(y, bias)

        return op.wrap_nested(y, "conv2d")


class Sequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = nn.ModuleList(list(modules))

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x
    

if __name__ == "__main__":
    import tvm 
    from tvm import relax
    import numpy as np 

    mlp = MLPBlock(32, 128)
    mod, params = mlp.export_tvm({"forward": {"x": nn.spec.Tensor((1, 4, 4, 32), "float32")}})
    mod.show()

    ex = tvm.compile(mod, target='cuda')
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(ex, dev)
    data = tvm.runtime.tensor(np.random.rand(1, 4, 4, 32).astype("float32"), dev)
    print(data)
    gpu_params = [
        tvm.runtime.tensor(np.random.rand(*p.shape).astype(p.dtype), dev) 
        for _, p in params
    ]
    gpu_out = vm["forward"](data, *gpu_params).numpy()
    print(gpu_out)