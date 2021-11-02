import torch
import numpy as np

def test_tensor():
    #from data
    data = [[1,2], [3, 4]]
    x_data = torch.tensor(data)
    #numpy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    #from another tensor
    x_ones = torch.ones_like(x_data)
    print(f"ones tensor : \n {x_ones}\n")
    x_rand = torch.rand_like(x_data, dtype=torch.float)

    #random or constant values
    shape = (2, 3, )
    rand_tensotr = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"random tensor: \n{rand_tensotr}\n")
    print(f"ones tensor: \n{ones_tensor}\n")
    print(f"zeros tensor \n{zeros_tensor}\n")

#attributes of a tensor
def test_tensor_attribute():
    tensor = torch.rand(3, 4)
    print(f"shape of tensor :{tensor.shape}")
    print(f"Datatype of type {tensor.dtype}")
    print(f"device tensor is stored {tensor.device}")

#operate on tensor
def test_operate_tensor():
    tensor = torch.rand(3, 4)
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')

#standard numpy-like indexing and slicings
def test_like_numpy():
    tensor = torch.ones(4, 4)
    print('first row:', tensor[0])
    print('first column: ', tensor[:, 0])
    print('last column；', tensor[..., -1])
    tensor[:, 1] = 0
    print(tensor)

#joining tensor
def test_torch_cat():
    tensor = torch.rand(4, 4)
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

#arithmetic operations
def test_arithmetic():
    tensor = torch.rand(4,4)
    y1 = tensor @ tensor.T
    y2  = tensor.matmul(tensor.T)

    y3 = torch.ones_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)

    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)

    #add
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg_item, type(agg_item))

    #in_place operation
    print(tensor, "\n")
    tensor.add_(5)
    print(tensor)

def test_bridge_numpy():
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n :{n}")
    t.add_(1)

    n = np.ones(5)
    t = torch.from_numpy(n)

    np.add(n, 1, out=n)