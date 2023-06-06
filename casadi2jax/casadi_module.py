import re

import casadi as cs

_look_up = {
    "@": "var",
    "log": "jnp.log",
    "exp": "jnp.exp",
    "sqrt": "jnp.sqrt",
    "cos": "jnp.cos",
    "acos": "jnp.arccos",
    "sin": "jnp.sin",
    "asin": "jnp.arcsin",
    "tan": "jnp.tan",
    "atan": "jnp.arctan",
    "atan2": "jnp.arctan2",
    "cosh": "jnp.cosh",
    "acosh": "jnp.arccosh",
    "sinh": "jnp.sinh",
    "asinh": "jnp.arcsinh",
    "tanh": "jnp.tanh",
    "atanh": "jnp.arctanh",
    "sq(": "jnp.square(",
}


def process_jax_expession(f, input_names, *args):
    if 'sparse' in f.__str__():
        text = f.__str__()
        input_names, text_result = sparse_expession_2_text(text, input_names, *args)
    else: 
        text = f.__str__()

        input_names, text_result = expession_2_text(text, input_names, *args)

    return input_names, text_result

def sparse_expession_2_text(expression, input_names, *args):
    for key, value in _look_up.items():
        expression = expression.replace(key, value)
    expression = expression.replace(",", "")

    for i in range(len(input_names) - 1, -1, -1):
        print(input_names[i])
        print(args[i].shape)
        for j in range(args[i].shape[0], -1, -1):
            expression = expression.replace(
                input_names[i] + "_" + str(j), f"{input_names[i]}[" + str(j) + "]"
            )
    # read expression line by line  
    expression_multi_line = expression.split("\n")
    result_expression = []
    
    for i, line in enumerate(expression_multi_line):
        # line = expression_multi_line[i]
        if "sparse" in line:
            line_parts = line.split(" ")[1]
            output_sizes = line_parts.split("-by-")
            output_array = f"\toutput = jnp.zeros(({output_sizes[0]}, {output_sizes[1]}))"
            result_expression.append(output_array)
            continue
        # of line starts with var append line to result expression 
        if "=" in line:
            if line.split("= ")[0].strip().startswith("var"):
                result_expression.append('\t'+line.replace(" ", ""))
                continue
        if '->' in line:
            line = line.split('->')
            output = f"\toutput = output.at{line[0].replace(' (', '[').replace(') ',']').replace(' ',',')}.set({line[1]})"
            result_expression.append(output)
        

    result_expression.append('\treturn output\n')
    return input_names, result_expression



def expession_2_text(expression, input_names, *args):
    for key, value in _look_up.items():
        expression = expression.replace(key, value)

    expression = expression.replace("[", "return jnp.array([").replace("]", "])")
    for i in range(len(input_names) - 1, -1, -1):
        print(input_names[i])
        print(args[i].shape)
        for j in range(args[i].shape[0], -1, -1):
            expression = expression.replace(
                input_names[i] + "_" + str(j), f"{input_names[i]}[" + str(j) + "]"
            )
    expression_multi_line = expression.split("return ")
    expression_result = [
        "\t" + line
        for line in expression_multi_line[0].replace(", ", "\n").split("\n")
        if line.strip() != ""
    ]
    expression_result.append("\treturn " + expression_multi_line[1])

    return input_names, expression_result

def process_jax_function(f, *args):
    input_names = []

    for element in args:
        if element.shape != (1, 1):
            if element.shape[1] != 1:
                raise ValueError("Only 1D arrays are supported")
            input_names.append(element[0].name().split("_0")[0])
    
    
    text = (f(*args)).__str__()


    input_names, text_result = process_jax_expession(text, input_names, *args)

    return input_names, text_result


def generate_jax_function(f, *args, **kwargs):
    if 'out_dir' in kwargs:
        out_dir = kwargs['out_dir'] + '/'
        kwargs.pop('out_dir')
    else:
        out_dir = ''
    if type(f)  == cs.casadi.Function:
        input_names, text_result = process_jax_function(f, *args)
    elif type(f) == cs.casadi.SX or type(f) == cs.casadi.MX:
        input_names = []
        for element in args:
            if element.shape != (1, 1):
                if element.shape[1] != 1:
                    raise ValueError("Only 1D arrays are supported")
                input_names.append(element[0].name().split("_0")[0])
        input_names, text_result = process_jax_expession(f, input_names, *args)
    
    with open(f"{out_dir}{f.name()}_jax.py", "w") as file:
        file.write("import jax.numpy as jnp\n")
        file.write("import numpy as np\n")
        file.write("import jax\n")
        file.write("@jax.jit\n")
        file.write(f"def {f.name()}(")

        for i in range(len(input_names)):
            file.write(f"{input_names[i]}")
            if i != len(input_names) - 1:
                file.write(", ")

        file.write("):\n")
        file.write("\n".join(text_result))


def generate_jax_function_params(f, *args):
    # TODO: add support for sparse functions
    input_names, text_result = process_jax_function(f, *args)
    params = []
    numeric_const_pattern = (
        "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    )

    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    for line in text_result:
        sub_line = line

        # remove all var0, var1, var2, etc.
        if len(sub_line.split("=")) == 2:
            sub_line = (
                sub_line.split("=")[0]
                + "="
                + re.sub(r"var[0-9]*", "", sub_line.split("=")[1])
            )
            sub_line = (
                sub_line.split("=")[0]
                + "="
                + re.sub(r"\[[0-9]*\]", "", sub_line.split("=")[1])
            )

            floats = rx.findall(sub_line.split("=")[1])
            if len(floats) == 0:
                continue
            for float_number in floats:
                params.append(float(float_number))
                line_new = line.replace(float_number, f"params[{len(params)-1}]")
            text_result[text_result.index(line)] = line_new
        if len(sub_line.split("return")) == 2:
            # remove all var0 up to var999

            sub_line = (
                sub_line.split("return")[0]
                + "return"
                + re.sub(r"var[0-9]*", "", sub_line.split("return")[1])
            )
            print(sub_line)
            sub_line = (
                sub_line.split("return")[0]
                + "return"
                + re.sub(r"\[[0-9]*\]", "", sub_line.split("return")[1])
            )

            floats = rx.findall(sub_line.split("return")[1])
            print(floats)
            if len(floats) == 0:
                continue
            ints = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            for float_number in floats:
                if float_number in ints:
                    params.append(float(float_number))
                    line_new = line.replace(
                        " " + float_number, f"params[{len(params)-1}]"
                    )
                    continue
                params.append(float(float_number))
                line_new = line.replace(float_number, f"params[{len(params)-1}]")
            text_result[text_result.index(line)] = line_new

    with open(f"{f.name()}_jax_trainable.py", "w") as file:
        file.write("import jax.numpy as jnp\n")
        file.write("import numpy as np\n")
        file.write("import jax\n")
        file.write("@jax.jit\n")
        file.write(f"def {f.name()}(")

        for i in range(len(input_names)):
            file.write(f"{input_names[i]}")
            file.write(", ")
        file.write("params):\n")
        file.write("\n".join(text_result))

    return params
