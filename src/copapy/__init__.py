# import pkgutil
from typing import Generator, Iterable, Any
from . import binwrite as binw
from .stencil_db import stencil_database

Operand = type['Net'] | float | int


def get_var_name(var: Any, scope: dict[str, Any] = globals()) -> list[str]:
    return [name for name, value in scope.items() if value is var]

# _ccode = pkgutil.get_data(__name__, 'stencils.c')
# assert _ccode is not None


sdb = stencil_database('src/copapy/obj/stencils_x86_64_O3.o')


class Node:
    def __init__(self) -> None:
        self.args: list[Net] = []
        self.name: str = ''

    def __repr__(self) -> str:
        #return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else self.value})"
        return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else (self.value if isinstance(self, Const) else '')})"


class Device():
    pass


class Net:
    def __init__(self, dtype: str, source: Node):
        self.dtype = dtype
        self.source = source

    def __mul__(self, other: Any) -> 'Net':
        return _add_op('mul', [self, other], True)

    def __rmul__(self, other: Any) -> 'Net':
        return _add_op('mul', [self, other], True)

    def __add__(self, other: Any) -> 'Net':
        return _add_op('add', [self, other], True)

    def __radd__(self, other: Any) -> 'Net':
        return _add_op('add', [self, other], True)

    def __sub__(self, other: Any) -> 'Net':
        return _add_op('sub', [self, other])

    def __rsub__(self, other: Any) -> 'Net':
        return _add_op('sub', [other, self])

    def __truediv__(self, other: Any) -> 'Net':
        return _add_op('div', [self, other])

    def __rtruediv__(self, other: Any) -> 'Net':
        return _add_op('div', [other, self])

    def __repr__(self) -> str:
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
        self.name: str = typed_op_name
        self.args: list[Net] = args


def _add_op(op: str, args: list[Any], commutative: bool = False) -> Net:
    arg_nets = [a if isinstance(a, Net) else const(a) for a in args]

    if commutative:
        arg_nets = sorted(arg_nets, key=lambda a: a.dtype)

    typed_op = '_'.join([op] + [a.dtype for a in arg_nets])

    if typed_op not in sdb.function_definitions:
        raise ValueError(f"Unsupported operand type(s) for {op}: {' and '.join([a.dtype for a in arg_nets])}")

    result_type = sdb.function_definitions[typed_op].split('_')[0]

    result_net = Net(result_type, Op(typed_op, arg_nets))

    return result_net


#def read_input(hw: Device, test_value: float):
#    return Net(type(value))


def const(value: Any) -> Net:
    assert isinstance(value, (int, float, bool)), f'Unsupported type for const: {type(value).__name__}'
    new_const = Const(value)
    return Net(new_const.dtype, new_const)


def _get_data_and_dtype(value: Any) -> tuple[str, float | int]:
    if isinstance(value, int):
        return ('int', int(value))
    elif isinstance(value, float):
        return ('float', float(value))
    elif isinstance(value, bool):
        return ('bool', int(value))
    else:
        raise ValueError(f'Non supported data type: {type(value).__name__}')


class vec3d:
    def __init__(self, value: tuple[Net, Net, Net]):
        self.value = value

    def __add__(self, other: 'vec3d') -> 'vec3d':
        a1, a2, a3 = self.value
        b1, b2, b3 = other.value
        return vec3d((a1 + b1, a2 + b2, a3 + b3))


def const_vector3d(x: float, y: float, z: float) -> vec3d:
    return vec3d((const(x), const(y), const(z)))


def get_path_segments(root: Iterable[Node]) -> Generator[list[Node], None, None]:
    """List of all possible paths. Ops in order of execution (output at the end)
    """
    def rekursiv_node_search(node_list: Iterable[Node], path: list[Node]) -> Generator[list[Node], None, None]:
        for node in node_list:
            new_path = [node] + path
            if node.args:
                yield from rekursiv_node_search([net.source for net in node.args], new_path)
            else:
                yield new_path

    known_nodes: set[Node] = set()
    sorted_path_list = sorted(rekursiv_node_search(root, []), key=lambda x: -len(x))

    for path in sorted_path_list:
        sflag = False
        for i, net in enumerate(path):
            if net in known_nodes or i == len(path) - 1:
                if sflag:
                    if i > 0:
                        yield path[:i+1]
                    break
            else:
                sflag = True
                known_nodes.add(net)


def get_ordered_ops(path_segments: list[list[Node]]) -> Generator[Node, None, None]:
    """Merge in all tree branches at branch position into the path segments
    """
    finished_paths: set[int] = set()

    for i, path in enumerate(path_segments):
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


def get_consts(op_list: list[Node]) -> list[tuple[str, Net, float | int]]:
    """Get all const nodes in the op list

    Returns:
        List of tuples of (name, net, value)"""
    net_lookup = {net.source: net for op in op_list for net in op.args}
    return [(n.name, net_lookup[n], n.value) for n in op_list if isinstance(n, Const)]


def add_read_ops(node_list: list[Node]) -> Generator[tuple[Net | None, Node], None, None]:
    """Add read operation before each op where arguments are not already positioned
    correctly in the registers

    Returns:
        Yields tuples of a net and a operation. The net is only provided
        for new added read operations. Otherwise None is returned in the tuple."""
    registers: list[None | Net] = [None] * 2

    # Generate result net lookup table
    net_lookup = {net.source: net for node in node_list for net in node.args}

    for node in node_list:
        if not node.name.startswith('const_'):
            for i, net in enumerate(node.args):
                if net != registers[i]:
                    #if net in registers:
                    #    print('x  swap registers')
                    type_list = ['int' if r is None else r.dtype for r in registers]
                    print(type_list)
                    new_node = Op(f"read_{net.dtype}_reg{i}_" + '_'.join(type_list), [])
                    yield net, new_node
                    registers[i] = net

            if node in net_lookup:
                yield None, node
                registers[0] = net_lookup[node]
            else:
                print('--->', node)
                yield None, node


def add_write_ops(net_node_list: list[tuple[Net | None, Node]], const_list: list[tuple[str, Net, float | int]]) -> Generator[tuple[Net | None, Node], None, None]:
    """Add write operation for each new defined net if a read operation is later followed"""

    # Initialize set of nets with constants
    stored_nets = {c[1] for c in const_list}

    assert all(node.name.startswith('read_') for net, node in net_node_list if net)
    read_back_nets = {net for net, _ in net_node_list if net}

    for net, node in net_node_list:
        if isinstance(node, Write):
            yield node.args[0], node
        else:
            yield net, node

        if net and net in read_back_nets and net not in stored_nets:
            yield net, Write(net)
            stored_nets.add(net)


def get_nets(*inputs: Iterable[Iterable[Any]]) -> list[Net]:
    nets: set[Net] = set()

    for input in inputs:
        for el in input:
            for net in el:
                if isinstance(net, Net):
                    nets.add(net)

    return list(nets)


def compile_to_instruction_list(end_nodes: Iterable[Node] | Node) -> binw.data_writer:
    if isinstance(end_nodes, Node):
        node_list = [end_nodes]
    else:
        node_list = list(end_nodes)

    path_segments = list(get_path_segments(node_list))
    ordered_ops = list(get_ordered_ops(path_segments))
    const_list = get_consts(ordered_ops)
    output_ops = list(add_read_ops(ordered_ops))
    extended_output_ops = list(add_write_ops(output_ops, const_list))

    for net, node in extended_output_ops:
        print(node.name)

    # Get all nets associated with heap memory
    variable_list = get_nets(const_list, extended_output_ops)

    dw = binw.data_writer(sdb.byteorder)

    def variable_mem_layout(variable_list: list[Net]) -> tuple[list[tuple[Net, int, int]], int]:
        offset: int = 0
        object_list: list[tuple[Net, int, int]] = []

        for variable in variable_list:
            lengths = sdb.var_size['dummy_' + variable.dtype]
            object_list.append((variable, offset, lengths))
            offset += (lengths + 3) // 4 * 4

        return object_list, offset

    object_list, data_section_lengths = variable_mem_layout(variable_list)

    # Write data
    dw.write_com(binw.Command.ALLOCATE_DATA)
    dw.write_int(data_section_lengths)

    for net, out_offs, lengths in object_list:
        dw.add_variable(net, out_offs, lengths, net.dtype)
        if isinstance(net.source, Const):
            dw.write_com(binw.Command.COPY_DATA)
            dw.write_int(out_offs)
            dw.write_int(lengths)
            dw.write_value(net.source.value, lengths)
            print(f'+ {net.dtype} {net.source.value}')

    # write auxiliary_functions
    # TODO

    # Prepare program code and relocations
    object_addr_lookp = {net: out_offs for net, out_offs, _ in object_list}
    data_list: list[bytes] = []
    patch_list: list[tuple[int, int, int]] = []
    offset = 0  # offset in generated code chunk

    print('object_addr_lookp: ', object_addr_lookp)

    data = sdb.get_func_data('function_start')
    data_list.append(data)
    offset += len(data)

    for result_net, node in extended_output_ops:
        assert node.name in sdb.function_definitions, f"- Warning: {node.name} prototype not found"
        data = sdb.get_func_data(node.name)
        data_list.append(data)
        print(f"* {node.name} ({offset}) " + ' '.join(f'{d:02X}' for d in data))

        for patch in sdb.get_patch_positions(node.name):
            assert result_net, f"Relocation found but no net defined for operation {node.name}"
            object_addr = object_addr_lookp[result_net]
            patch_value = object_addr + patch.addend - (offset + patch.addr)
            print('patch: ', patch, object_addr, patch_value)
            patch_list.append((patch.type.value, offset + patch.addr, patch_value))

        offset += len(data)

    data = sdb.get_func_data('function_end')
    data_list.append(data)
    offset += len(data)
    print('function_end', offset, data)

    # allocate program data
    dw.write_com(binw.Command.ALLOCATE_CODE)
    dw.write_int(offset)

    # write program data
    dw.write_com(binw.Command.COPY_CODE)
    dw.write_int(0)
    dw.write_int(offset)
    dw.write_bytes(b''.join(data_list))

    # write relocations
    for patch_type, patch_addr, object_addr in patch_list:
        dw.write_com(binw.Command.PATCH_OBJECT)
        dw.write_int(patch_addr)
        dw.write_int(patch_type)
        dw.write_int(object_addr, signed=True)

    # set entry point
    dw.write_com(binw.Command.SET_ENTR_POINT)
    dw.write_int(0)

    return dw


def read_variable(bw: binw.data_writer, net: Net) -> None:
    assert net in bw.variables, f"Variable {net} not found in data writer variables"
    addr, lengths, _ = bw.variables[net]
    bw.write_com(binw.Command.READ_DATA)
    bw.write_int(addr)
    bw.write_int(lengths)
