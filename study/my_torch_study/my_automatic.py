import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entorpy_with_logits(z, y)

#tensors function and computational graph
print('Gradient functional for z=', z.grad_fn)
print('gradient functional for loss = ', loss.grad_fn)

#computing gradients
loss.backward()
print(w.grad)
print(b.grad)

#disabling gradient trancking
def test_gradient():
    z = torch.matmul(x, w) + b
    print(z.require_grad)
    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.require_grad)

    z = torch.matmul(x, w) + b
    z_det = z.detach()
    print(z_det.require_grad)



#optional reading tensor gradient and jacobian products
def test_jacobian():
    inp = torch.eye(5, requires_grad=True)
    out = (inp+1).pow(2)
    out.backward(torch.ones_like(inp), retain_graph = True)
    print("first call \n", inp.grad)
    out.backward(torch.ones_like(inp), retain_graph = True)
    print("\nsecond call \n", inp.grad)
    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_graph = True)
    print("\n call after zeroing gradient \n", inp.grad)