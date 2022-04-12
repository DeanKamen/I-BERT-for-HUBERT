import numpy as np
# import torch


def ndarray_to_str(x, brkt_op='{', brkt_cl='}', delim=','):
    ndims = len(x.shape)
    nelem = x.shape[0]
    s = brkt_op
    for i in range(nelem):
        if ndims <= 1:
            s += str(int(x[i])) #THIS INT RIGHT HERE: forces to print without decimal place. sometimes necessary.
            #s += str(x[i])
            if i < nelem - 1:
                s += delim + ' '
        else:
            s += ndarray_to_str(x[i], brkt_op='', brkt_cl='')
            if i < nelem - 1:
                s += delim + '\n'
    s += brkt_cl
    return s


def ndarray_to_var(x, vartype, varname):
    body_str = ndarray_to_str(x)

    s = "\n{} {}".format(vartype, varname)
    arr_size = 1
    for z in np.shape(x):
        arr_size *= z
    s = s + "[{}]".format(arr_size)
    s = s + " = " + body_str + ";\n"
    return s


def var_to_var(x, vartype, varname):
    body_str = str(x)

    s = "\n{} {}".format(vartype, varname)
    s = s + " = " + body_str + ";\n"
    return s


def h_header(filename):
    s = '#ifndef {}\n'.format(filename.upper().replace('.', '_'))
    s += '#define {}\n'.format(filename.upper().replace('.', '_'))
    return s


def h_footer(filename):
    s = '\n#endif // {}\n'.format(filename.upper().replace('.', '_'))
    return s


def export_header(npa, filename):
    header = h_header(filename+".h")
    s = ndarray_to_var(npa, 'const int', filename)
    footer = h_footer(filename+".h")

    with open(filename+".h", 'w') as f:
        f.write(header)
        f.write(s)
        f.write(footer)

