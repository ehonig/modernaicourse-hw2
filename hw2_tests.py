import math
import torch
import mugrade


class Variable:
    def __init__(self, value, function = None, parents=None):
        """
        Initialize the variable with its needed properties.
        """
        self.value = value
        self.grad = None
        self.function = function
        self.parents = parents
        self.num_children = 0

    def __repr__(self): return f"Variable({self.value}, grad={self.grad})"
    def __add__(self, other): return Add()(self, other)
    def __sub__(self, other): return Subtract()(self, other)
    def __mul__(self, other): return Multiply()(self, other)
    def __truediv__(self, other): return Divide()(self, other)
    def __neg__(self): return Negate()(self)
    def __pow__(self, d): return Power(d)(self)
    def log(self): return Log()(self)
    def exp(self): return Exp()(self)


class Function:
    def __call__(self, *args):
        value = self.forward(*[a.value for a in args])
        for p in args:
            p.num_children += 1
        return Variable(value, function=self, parents=args)


class Multiply(Function):
    def forward(self, x, y):
        return x * y

    def backward(self, grad, x, y):
        return [y * grad, x * grad]


class Negate(Function):
    def forward(self, x):
        return -x

    def backward(self, grad, x):
        return [-grad]

def test_Add(Add):
    f = Add()
    assert(f.forward(2.5, -0.5) == 2.0)
    g = f.backward(3.0, 2.5, -0.5)
    assert(len(g) == 2)
    assert(g[0] == 3.0 and g[1] == 3.0)


def submit_Add(Add):
    f = Add()
    mugrade.submit(f.forward(-1.5, 3.0))
    mugrade.submit(list(f.backward(2.5, -1.5, 3.0)))


def test_Subtract(Subtract):
    f = Subtract()
    assert(f.forward(2.5, -0.5) == 3.0)
    g = f.backward(3.0, 2.5, -0.5)
    assert(len(g) == 2)
    assert(g[0] == 3.0 and g[1] == -3.0)


def submit_Subtract(Subtract):
    f = Subtract()
    mugrade.submit(f.forward(-1.5, 3.0))
    mugrade.submit(list(f.backward(2.5, -1.5, 3.0)))


def test_Divide(Divide):
    f = Divide()
    assert(f.forward(9.0, 3.0) == 3.0)
    g = f.backward(4.0, 9.0, 3.0)
    assert(len(g) == 2)
    assert(math.isclose(g[0], 4.0 / 3.0))
    assert(math.isclose(g[1], -4.0))


def submit_Divide(Divide):
    f = Divide()
    mugrade.submit(f.forward(-8.0, 2.0))
    mugrade.submit(list(f.backward(1.5, -8.0, 2.0)))


def test_Power(Power):
    f = Power(3)
    assert(f.forward(2.0) == 8.0)
    g = f.backward(2.0, 2.0)
    assert(len(g) == 1)
    assert(g[0] == 24.0)

    f0 = Power(0)
    assert(f0.forward(5.0) == 1.0)
    g0 = f0.backward(7.0, 5.0)
    assert(len(g0) == 1)
    assert(g0[0] == 0.0)


def submit_Power(Power):
    f = Power(3)
    f0 = Power(0)
    mugrade.submit(f.forward(-2.0))
    mugrade.submit(list(f.backward(1.5, -2.0)))
    mugrade.submit(list(f0.backward(7.0, 5.0)))


def test_Log(Log):
    f = Log()
    assert(math.isclose(f.forward(math.e), 1.0))
    g = f.backward(2.0, 4.0)
    assert(len(g) == 1)
    assert(math.isclose(g[0], 0.5))


def submit_Log(Log):
    f = Log()
    mugrade.submit(f.forward(3.5))
    mugrade.submit(f.backward(2.0, 3.5)[0])


def test_Exp(Exp):
    f = Exp()
    assert(math.isclose(f.forward(1.0), math.e))
    g = f.backward(2.0, 1.0)
    assert(len(g) == 1)
    assert(math.isclose(g[0], 2.0 * math.e))


def submit_Exp(Exp):
    f = Exp()
    mugrade.submit(f.forward(-0.5))
    mugrade.submit(f.backward(2.0, -0.5)[0])


def test_compute_gradients(compute_gradients):
    Variable.compute_gradients = compute_gradients

    x = Variable(3.0)
    y = Variable(4.0)
    z = ((-(x * y) * x * x) * (-y))
    z.compute_gradients()
    assert(z.grad == 1.0)
    assert(math.isclose(x.grad, 432.0))
    assert(math.isclose(y.grad, 216.0))

    w = Variable(-2.0)
    w.compute_gradients()
    assert(w.grad == 1.0)


def submit_compute_gradients(compute_gradients):
    Variable.compute_gradients = compute_gradients
    x = Variable(1.5)
    y = Variable(-2.0)
    a = x * y * (-x) * (-y)
    z = -(a * a)
    z.compute_gradients()
    mugrade.submit(a.grad)
    mugrade.submit(x.grad)
    mugrade.submit(y.grad)


def test_cross_entropy_loss(cross_entropy_loss):
    y_pred = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    y = torch.tensor([0, 2])
    loss = cross_entropy_loss(y_pred, y)
    assert(loss.ndim == 0)
    assert(torch.isclose(loss, torch.tensor(0.9076060056686401), atol=1e-6))


def submit_cross_entropy_loss(cross_entropy_loss):
    y_pred = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0], [-1.0, 2.0, 1.0]])
    y = torch.tensor([0, 2, 1])
    loss = cross_entropy_loss(y_pred, y)
    mugrade.submit(loss.item())
    mugrade.submit(type(loss))


def test_error(error):
    y_pred = torch.tensor([[3.0, 1.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]])
    y = torch.tensor([0, 1, 1, 1])
    err = error(y_pred, y)
    assert(err.ndim == 0)
    assert(torch.allclose(err, torch.tensor(0.25)))


def submit_error(error):
    y_pred = torch.tensor([[3.0, 1.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]])
    y = torch.tensor([0, 1, 1, 1])
    mugrade.submit(error(y_pred, y).item())
    mugrade.submit(type(error(y_pred, y)))


def test_train_sgd(train_sgd):
    X = torch.tensor([[2.0, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]])
    y = torch.tensor([1, 1, 0, 0])

    W_small = train_sgd(X, y, epochs=1, step_size=0.1, batch_size=2)
    W_expected = torch.tensor([[-0.13340412, -0.13340412], [0.13340412, 0.13340412]])
    assert(torch.allclose(W_small, W_expected, atol=1e-6))

    W = train_sgd(X, y, epochs=20, step_size=0.1, batch_size=2)
    assert(isinstance(W, torch.Tensor))
    assert(W.shape == (2, 2))
    pred = (X @ W.T).argmax(dim=-1)
    assert(torch.equal(pred, y))
    assert(not torch.allclose(W, torch.zeros_like(W)))


def submit_train_sgd(train_sgd):
    X = torch.tensor([[2.0, 1.0], [1.0, 2.0], [-2.0, -1.0], [-1.0, -2.0]])
    y = torch.tensor([1, 1, 0, 0])

    W = train_sgd(X, y, epochs=15, step_size=0.1, batch_size=2)
    mugrade.submit(W.detach().numpy())
    mugrade.submit((X @ W.T).argmax(dim=-1).numpy())
    mugrade.submit(W.norm().item())
