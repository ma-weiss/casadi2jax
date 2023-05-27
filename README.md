<h1 align="center">casadi2jax</h1>

Turn CasADi expressions into trainable JAX expressions. The output will be a python file that can be imported and used as a JAX function. There are two option to do this:
- with parameters: the parameters can be updated via gradient descent
- without parameters: the parameters are fixed and the output is a pure JAX function to speed up the computation

Optimise your symbolic expressions via gradient descent!

## Installation

```bash
pip3 install -e .
```

usage:

to generate a python file containing the jax function use:

```python 
import casadi2jax as c2j
import casadi as cs 

x = cs.SX.sym('x')
exp = cs.sin(x) + 1

c2j.generate_jax_function(exp, x)

```

or 

```python 
import casadi2jax as c2j
import casadi as cs 

x = cs.SX.sym('x')
exp = cs.sin(x) + 1

func = cs.Function("f", [x], [exp])

# convert function back to method

exp_new = func(x)

c2j.generate_jax_function(exp_new, x)

```

to generate a Jax function with tunable parameters:

```python 
import casadi2jax as c2j
import casadi as cs 

x = cs.SX.sym('x')
exp = cs.sin(x) + 1

params = c2j.generate_jax_function_params(exp, x)

```

