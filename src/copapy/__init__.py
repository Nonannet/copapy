import re
import pkgutil

def get_var_name(var, scope=globals()):
    return [name for name, value in scope.items() if value is var]

def _get_c_function_definitions(code: str):
    ret = re.findall(r".*?void\s+([a-z_1-9]*)\s*\([^\)]*?\)[^\}]*?\{[^\}]*?result_([a-z_]*)\(.*?", code, flags=re.S)
    return {r[0]: r[1] for r in ret}

_function_definitions = _get_c_function_definitions(pkgutil.get_data(__name__, 'ops.c').decode('utf-8'))

class Node:
    def __repr__(self):
        #return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else self.value})"
        return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else (self.value if isinstance(self, Const) else '')})"

class Device():
    pass
         
class Net:
    def __init__(self, dtype: str, source: Node):
        self.dtype = dtype
        self.source = source

    def __mul__(self, other):
        return _add_op('mul', [self, other], True)

    def __rmul__(self, other):
        return _add_op('mul', [self, other], True)
    
    def __add__(self, other):
        return _add_op('add', [self, other], True)

    def __radd__(self, other):
        return _add_op('add', [self, other], True)

    def __sub__ (self, other):
        return _add_op('sub', [self, other])

    def __rsub__ (self, other):
        return _add_op('sub', [other, self])

    def __truediv__ (self, other):
        return _add_op('div', [self, other])

    def __rtruediv__ (self, other):
        return _add_op('div', [other, self])

    def __repr__(self):
        names = get_var_name(self)
        return f"{'name:' + names[0] if names else 'id:' + str(id(self))[-5:]}"


class Const(Node):
    def __init__(self, value: float | int | bool):
        self.dtype, self.value = _get_data_and_dtype(value)
        self.name = 'const_' + self.dtype

        #if self.name not in _function_definitions:
        #    raise ValueError(f"Unsupported operand type for a const: {self.dtype}")

        self.args = []

class Write(Node):
    def __init__(self, net: Net):
        self.name = 'write_' + net.dtype
        self.args = [net]

        #if self.name not in _function_definitions:
        #    raise ValueError(f"Unsupported operand type for write: {net.dtype}")

class Op(Node):
    def __init__(self, typed_op_name: str, args: list[Net]):
        assert not args or any(isinstance(t, Net) for t in args), 'args parameter must be of type list[Net]'
        self.name = typed_op_name
        self.args = args

def _add_op(op: str, args: list[Net], commutative = False):
    args = [a if isinstance(a, Net) else const(a) for a in args]

    if commutative:
        args = sorted(args, key=lambda a: a.dtype)

    typed_op =  '_'.join([op] + [a.dtype for a in args])

    if typed_op not in _function_definitions:
        raise ValueError(f"Unsupported operand type(s) for {op}: {' and '.join([a.dtype for a in args])}")

    result_type = _function_definitions[typed_op]

    result_net = Net(result_type, Op(typed_op, args))

    return result_net

#def read_input(hw: Device, test_value: float):
#    return Net(type(value))

def const(value: float | int | bool):
    new_const = Const(value)
    return Net(new_const.dtype, new_const)

def _get_data_and_dtype(value):
    if isinstance(value, int):
        return ('int', int(value))
    elif isinstance(value, float):
        return ('float', float(value))
    elif isinstance(value, bool):
        return ('bool', int(value))
    else:
        raise ValueError(f'Non supported data type: {type(value).__name__}')

def const_vector3d(x: float, y: float, z: float):
    return vec3d((const(x), const(y), const(z)))

class vec3d:
    def __init__(self, value: tuple[float, float, float]):
        self.value = value

    def __add__(self, other):
        return vec3d(tuple(a+b for a,b in zip(self.value, other.value)))


def get_multiuse_nets(root: list[Op]):
    """
    Finds all nets that get accessed more than one time. Therefore
    storage on the heap might be better.
    """
    known_nets: set[Net] = set()

    def recursiv_node_search(net_list: list[Net]):
        for net in net_list:
            #print(net)
            if net in known_nets:
                yield net
            else:
                known_nets.add(net)
                yield from recursiv_node_search(net.source.args)

    return set(recursiv_node_search(op.args[0] for op in root))


def get_path_segments(root: list[Op]) -> list[list[Op]]:
    """
    List of all possible paths. Ops in order of execution (output at the end)
    """
    def recursiv_node_search(op_list: list[Op], path: list[Op]) -> list[Op]:
        for op in op_list:
            new_path = [op] + path
            if op.args:
                yield from recursiv_node_search([net.source for net in op.args], new_path)
            else:
                yield new_path

    known_ops: set[Op] = set()
    sorted_path_list = sorted(recursiv_node_search(root, []), key=lambda x: -len(x))

    for path in sorted_path_list:
        sflag = False
        for i, net in enumerate(path):
            if net in known_ops or i == len(path) - 1:
                if sflag:
                    if i > 0:
                        yield path[:i+1]
                    break
            else:
                sflag = True
                known_ops.add(net)


def get_ordered_ops(path_segments: list[list[Op]]) -> list[Op]:
    finished_paths: set[int] = set()
    
    for i, path in enumerate(path_segments):
        #print(i)
        if i not in finished_paths:
            for op in path:
                for j in range(i + 1, len(path_segments)):
                    path_stub = path_segments[j]
                    if op == path_stub[-1]:
                        for insert_op in path_stub[:-1]:
                            #print('->', insert_op)
                            yield insert_op
                        finished_paths.add(j)
                #print('- ', op)
                yield op
            finished_paths.add(i)

                               
def get_consts(op_list: list[Node]):
    net_lookup = {net.source: net for op in op_list for net in op.args}
    return [(n.name, net_lookup[n], n.value) for n in op_list if isinstance(n, Const)]


def add_read_ops(op_list):
    """Add read operation before each op where arguments are not allredy possitioned
    correctly in the registers
    
    Returns:
        Yields a tuples of a net and a operation. The net is the result net
        from the retuned operation"""
    registers = [None] * 16

    net_lookup = {net.source: net for op in op_list for net in op.args}
    
    for op in op_list:
        if not op.name.startswith('const_'):
            for i, net in enumerate(op.args):
                if net != registers[i]:
                    #if net in registers:
                    #    print('x  swap registers')
                    new_op = Op('read_reg' + str(i) + '_' + net.dtype, [])
                    yield net, new_op
                    registers[i] = net
    
            yield net_lookup.get(op), op
            if op in net_lookup:
                registers[0] = net_lookup[op]


def add_write_ops(op_list, const_list):
    """Add write operation for each new defined net if a read operation is later folowed"""
    stored_nets = {c[1] for c in const_list}
    read_back_nets = {net for net, op in op_list if op.name.startswith('read_reg')}
    
    for net, op in op_list:
        yield net, op
        if net in read_back_nets and net not in stored_nets:
            yield (net, Op('write_' + net.dtype, [net]))
            stored_nets.add(net)