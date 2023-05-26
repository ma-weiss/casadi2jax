<h1 align="center">casadi2jax</h1>

Turn CasADi expressions into trainable JAX expressions. The output will be a python file that can be imported and used as a JAX function. There are two option to do this:
- with parameters: the parameters can be updated via gradient descent
- without parameters: the parameters are fixed and the output is a pure JAX function to speed up the computation

Optimise your symbolic expressions via gradient descent!

## Installation

```bash
pip3 install -e .‚
```