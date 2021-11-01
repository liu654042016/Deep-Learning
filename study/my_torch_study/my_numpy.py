# -*- coding: utf-8 -*-
import numpy as np
import math
import random
import torch
def test_numpy(self):
    #create random input and output data
    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)

    #random initalize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    learning_rate = 1e-6
    for t in range(2000):
        #forward pass : compute predicted y
        #y = a + bx + cx^2 +dx^3
        y_pred = a+b*x+c*x**2+d*x**3

        loss = np.square(y_pred - y).sum()
        if t%100 == 99:
            print(t, loss)
        #backpiop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0*(y_pred-y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred*x).sum()
        grad_c = (grad_y_pred*x**2).sum()
        grad_d = (grad_y_pred*x**3).sum()

        #update weights
        a-= learning_rate * grad_a
        b-=learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
    print(f"Reslut: y={a}+{b}x+{c}^x+{d}x^3")

####################################torch_demo################################################
import torch
import math

def test_tensor():
    dtype = torch.float
    device = torch.device("cpu")
    #device = torch.device("cuda: 0")

    #create random input and output data
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(2000):
        y_pred = a + b*x + c*x**2 + d*x**3

        loss = (y_pred-y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

        grad_y_pred = 2*(y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred*x).sum()
        grad_c = (grad_y_pred*x**2).sum()
        grad_d = (grad_y_pred*x**3).sum()

        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_b
    print(f"restul: y={a.item()}+b{b.item()}x + c{c.item()}x^2 + {d.item()}x^3")


#autograd
def test_autograd():
    dtype = torch.float
    device = torch.device('device')

    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    a =torch.randn((), device = device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device = device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device= device, dtype = dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_late = 1e-6
    for t in enumerate(2000):
        y_pred = a = b*x**2 + c*x**3 + d*x**4

        loss = (y_pred - y).pow(2).sum()
        if t%100 == 99:
            print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            a -= learning_late * a.grad
            b -= learning_late * b.grad
            c -=learning_late * c.grad
            d -= learning_late * d.grad

            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

def test_new_autograd():
    class LegendrePolynomial3(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return 0.5*(5*input**3 - 3*input)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            return grad_output*1.5*(5*input**2 -1)
    dtype = torch.float
    device = torch.device("cpu")

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    b = torch.full((), -1.0, device = device, dtype=dtype,requires_grad=True)
    c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 5e-6
    for t in range(2000):
        P3 = LegendrePolynomial3.apply

        y_pred = a + b * P3(c+d*x)
        loss = (y_pred - y).pow(2).sum()

        loss.backward()

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

    print(f'result: y={a.item()} + {b.item()}*P3({c.item() + {d.item()}*x)')
#nn model
def test_Model():
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-6
    for t in range(2000):
        y_pred = model(xx)

        loss = loss_fn(y_pred, y)

        if t % 100 == 99:
            print(t, loss.item())

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    Linear_layer = model[0]

    print(f"result y={Linear_layer.bias.item()} + {Linear_layer.weight[: ,0].item()}X + {Linear_layer.weight[:, 1].item()}x^2"+{Linear_layer.weight[:, 2]}^3)

def torch_custom_model():
    class Polynomial3(torch.nn.Module):
        def __init__(self):
            super.__init__()
            self.a = torch.nn.Parameter(torch.randn(()))
            self.b = torch.nn.Parameter(torch.randn(()))
            self.c = torch.nn.Parameter(torch.randn(()))
            self.d = torch.nn.Parameter(torch.randn(()))
            self.e = torch.nn.Parameter(torch.randn(()))

        def forward(self, x):
            # y = self.a + self.b * x + self.c * x **2 + self.d * x **3
            # for exp in range(4, random.randint(4, 6 )):
            #     y = y + self.e * x ** exp
            # return y_pred
            return self.a + self.b * x + self.c * x ** 2 + self.d * x **3
        def string(self):
            return f'y={self.a.item()} + {self.b.item()}x + {self.c.item()}x^2 + {self.d.item()}x^3 + {self.e.item()}x^4 + {self.e.item()}x^5'

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    model = Polynomial3()

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6)

    for t in range(2000):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"result {model.string()}")

def test_weight_torch:
    class DynamicNet(torch.nn.Module):
        def __init__(self):
            super.__init__(self)
            self.a = torch.nn.Parameter(torch.randn(()))
            self.b = torch.nn.Parameter(torch.randn(()))
            self.c = torch.nn.Parameter(torch.randn(()))
            self.d = torch.nn.Parameter(torch.randn(()))
            self.e = torch.nn.Parameter(torch.randn(()))

        def forward(self, x):
            y = self.a + self.b * x + self.c * x ** 2 + self.d * x **3
            for exp in range(4, random.randint(4, 6)):
                y = y + self.e * x ** exp
            return y
        def string(self):
            return f'y = {self.a.item()} + {self.b.item()}x + {self.c.item()}x^2 + {self.d.item()}x^3 + {self.e.item()}x^4'

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    model = DynamicNet()

    criterion  = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-8, momentum=0.9)

    for t in range(30000):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if t % 2000 == 1999:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"result: {model.string()}")