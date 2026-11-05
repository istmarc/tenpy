# tenpy

Python bindings for the ten library (pythonizations)

>> [!NOTE]
The python bindings of ten doesn't support automatic differentiation and lazy evaluation.

## Examples

- Matrix multiplication

```python
import ten
a = ten.random.rand_norm((2,3))
b = ten.random.rand_unif((3,4))
c = a @ b
print(c)
```

- Random walk

```python
import ten
n = 100
s = ten.process.random_walk(n)
t,x = ten.process.continuous_random_walk(s, n//2, n)
print(t)
print(x)
```

- Linear model

```python
x = ten.random.rand_norm((4, 2))
y = ten.random.rand_norm([4])

model = ten.ml.linear_model()
model.fit(x, y)
beta = model.coef()
print(beta)
```

## Install

- Follow the instructions at [istmar.github.io/ten](istmarc.github.io/ten) to build and install ten from source
- Install tenpy using `pip install tenpy/`

