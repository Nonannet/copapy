# import pkgutil
from typing import Generator, Iterable, Any
from . import binwrite as binw
from .stencil_db import stencil_database
from collections import defaultdict, deque
from coparun_module import coparun, read_data_mem
import struct

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

    def __neg__(self) -> 'Net':
        return _add_op('sub', [const(0), self])

    def __gt__(self, other: Any) -> 'Net':
        return _add_op('gt', [self, other])

    def __lt__(self, other: Any) -> 'Net':
        return _add_op('gt', [other, self])

    def __eq__(self, other: Any) -> 'Net':
        return _add_op('eq', [self, other])

    def __mod__(self, other: Any) -> 'Net':
        return _add_op('mod', [self, other])

    def __rmod__(self, other: Any) -> 'Net':
        return _add_op('mod', [other, self])

    def __repr__(self) -> str:
        names = get_var_name(self)
        return f"{'name:' + names[0] if names else 'id:' + str(id(self))[-5:]}"

    def __hash__(self) -> int:
        return id(self)


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


def stable_toposort(edges: Iterable[tuple[Node, Node]]) -> list[Node]:
    """Perform a stable topological sort on a directed acyclic graph (DAG).
    Arguments:
        edges: Iterable of (u, v) pairs meaning u -> v

    Returns:
        List of nodes in topologically sorted order.
    """

    # Track adjacency and indegrees
    adj: defaultdict[Node, list[Node]] = defaultdict(list)
    indeg: defaultdict[Node, int] = defaultdict(int)
    order: dict[Node, int] = {}  # first-appearance order of each node

    # Build graph and order map
    pos = 0
    for u, v in edges:
        if u not in order:
            order[u] = pos
            pos += 1
        if v not in order:
            order[v] = pos
            pos += 1
        adj[u].append(v)
        indeg[v] += 1
        indeg.setdefault(u, 0)

    # Initialize queue with nodes of indegree 0, sorted by first appearance
    queue = deque(sorted([n for n in indeg if indeg[n] == 0], key=lambda x: order[x]))
    result: list[Node] = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for nei in adj[node]:
            indeg[nei] -= 1
            if indeg[nei] == 0:
                queue.append(nei)

        # Maintain stability: sort queue by appearance order
        queue = deque(sorted(queue, key=lambda x: order[x]))

    # Check if graph had a cycle (not all nodes output)
    if len(result) != len(indeg):
        raise ValueError("Graph contains a cycle — topological sort not possible")

    return result


def get_all_dag_edges(nodes: Iterable[Node]) -> Generator[tuple[Node, Node], None, None]:
    """Get all edges in the DAG by traversing from the given nodes

    Arguments:
        nodes: Iterable of nodes to start the traversal from

    Yields:
        Tuples of (source_node, target_node) representing edges in the DAG
    """
    for node in nodes:
        yield from get_all_dag_edges(net.source for net in node.args)
        yield from ((net.source, node) for net in node.args)


def get_const_nets(nodes: list[Node]) -> list[Net]:
    """Get all nets with a constant nodes value

    Returns:
        List of nets whose source node is a Const
    """
    net_lookup = {net.source: net for node in nodes for net in node.args}
    return [net_lookup[node] for node in nodes if isinstance(node, Const)]


def add_read_ops(node_list: list[Node]) -> Generator[tuple[Net | None, Node], None, None]:
    """Add read node before each op where arguments are not already positioned
    correctly in the registers

    Arguments:
        node_list: List of nodes in the order of execution

    Returns:
        Yields tuples of a net and a node. The net is the result net
        for the node. If the node has no result net None is returned in the tuple.
    """

    registers: list[None | Net] = [None] * 2

    # Generate result net lookup table
    net_lookup = {net.source: net for node in node_list for net in node.args}

    for node in node_list:
        if not node.name.startswith('const_'):
            for i, net in enumerate(node.args):
                if id(net) != id(registers[i]):
                    #if net in registers:
                    #    print('x  swap registers')
                    type_list = ['int' if r is None else r.dtype for r in registers]
                    new_node = Op(f"read_{net.dtype}_reg{i}_" + '_'.join(type_list), [])
                    yield net, new_node
                    registers[i] = net

            if node in net_lookup:
                yield net_lookup[node], node
                registers[0] = net_lookup[node]
            else:
                yield None, node


def add_write_ops(net_node_list: list[tuple[Net | None, Node]], const_nets: list[Net]) -> Generator[tuple[Net | None, Node], None, None]:
    """Add write operation for each new defined net if a read operation is later followed

    Returns:
        Yields tuples of a net and a node. The associated net is provided for read and write nodes.
        Otherwise None is returned in the tuple.
    """

    # Initialize set of nets with constants
    stored_nets = set(const_nets)

    #assert all(node.name.startswith('read_') for net, node in net_node_list if net)
    read_back_nets = {
        net for net, node in net_node_list
        if net and node.name.startswith('read_')}

    for net, node in net_node_list:
        if isinstance(node, Write):
            yield node.args[0], node
        elif node.name.startswith('read_'):
            yield net, node
        else:
            yield None, node

        if net in read_back_nets and net not in stored_nets:
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


def compile_to_instruction_list(node_list: Iterable[Node], sdb: stencil_database) -> tuple[binw.data_writer, dict[Net, tuple[int, int, str]]]:
    variables: dict[Net, tuple[int, int, str]] = dict()
    
    ordered_ops = list(stable_toposort(get_all_dag_edges(node_list)))
    const_net_list = get_const_nets(ordered_ops)
    output_ops = list(add_read_ops(ordered_ops))
    extended_output_ops = list(add_write_ops(output_ops, const_net_list))

    # Get all nets associated with heap memory
    variable_list = get_nets([[const_net_list]], extended_output_ops)

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
        variables[net] = (out_offs, lengths, net.dtype)
        if isinstance(net.source, Const):
            dw.write_com(binw.Command.COPY_DATA)
            dw.write_int(out_offs)
            dw.write_int(lengths)
            dw.write_value(net.source.value, lengths)
            # print(f'+ {net.dtype} {net.source.value}')

    # write auxiliary_functions
    # TODO

    # Prepare program code and relocations
    object_addr_lookp = {net: out_offs for net, out_offs, _ in object_list}
    data_list: list[bytes] = []
    patch_list: list[tuple[int, int, int]] = []
    offset = 0  # offset in generated code chunk

    # print('object_addr_lookp: ', object_addr_lookp)

    data = sdb.get_func_data('function_start')
    data_list.append(data)
    offset += len(data)

    for associated_net, node in extended_output_ops:
        assert node.name in sdb.function_definitions, f"- Warning: {node.name} prototype not found"
        data = sdb.get_func_data(node.name)
        data_list.append(data)
        # print(f"* {node.name} ({offset}) " + ' '.join(f'{d:02X}' for d in data))

        for patch in sdb.get_patch_positions(node.name):
            assert associated_net, f"Relocation found but no net defined for operation {node.name}"
            object_addr = object_addr_lookp[associated_net]
            patch_value = object_addr + patch.addend - (offset + patch.addr)
            # print('patch: ', patch, object_addr, patch_value)
            patch_list.append((patch.type.value, offset + patch.addr, patch_value))

        offset += len(data)

    data = sdb.get_func_data('function_end')
    data_list.append(data)
    offset += len(data)
    # print('function_end', offset, data)

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

    return dw, variables


class Target():

    def __init__(self, arch: str = 'x86_64', optimization: str = 'O3') -> None:
        self.sdb = stencil_database(f"src/copapy/obj/stencils_{arch}_{optimization}.o")
        self._variables: dict[Net, tuple[int, int, str]] = dict()


    def compile(self, *variables: list[Net] | list[list[Net]]) -> None:
        nodes: list[Node] = []
        for s in variables:
            if isinstance(s, Net):
                nodes.append(Write(s))
            else:
                for net in s:
                    assert isinstance(net, Net)
                    nodes.append(Write(net))
                

        dw, self._variables = compile_to_instruction_list(nodes, self.sdb)
        dw.write_com(binw.Command.END_PROG)
        assert coparun(dw.get_data()) > 0


    def run(self) -> None:
        # set entry point and run code
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.SET_ENTR_POINT)
        dw.write_int(0)
        dw.write_com(binw.Command.END_PROG)
        assert coparun(dw.get_data()) > 0


    def read_value(self, net: Net) -> float | int:
        assert net in self._variables, f"Variable {net} not found"
        addr, lengths, var_type = self._variables[net]
        data = read_data_mem(addr, lengths)
        assert data is not None and len(data) == lengths, f"Failed to read variable {net}"
        en = {'little': '<', 'big': '>'}[self.sdb.byteorder]
        if var_type == 'float':
            if lengths == 4:
                value = struct.unpack(en + 'f', data)[0]
                assert isinstance(value, float)
                return value
            elif lengths == 8:
                value = struct.unpack(en + 'd', data)[0]
                assert isinstance(value, float)
                return value
            else:
                raise ValueError(f"Unsupported float length: {lengths}")
        elif var_type == 'int':
            if lengths in (1, 2, 4, 8):
                value = int.from_bytes(data, byteorder=self.sdb.byteorder, signed=True)
                assert isinstance(value, int)
                return value
            else:
                raise ValueError(f"Unsupported int length: {lengths}")
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

    def read_variable_remote(self, bw: binw.data_writer, net: Net) -> None:
        assert net in self._variables, f"Variable {net} not found in data writer variables"
        dw = binw.data_writer(self.sdb.byteorder)
        addr, lengths, _ = self._variables[net]
        bw.write_com(binw.Command.READ_DATA)
        bw.write_int(addr)
        bw.write_int(lengths)