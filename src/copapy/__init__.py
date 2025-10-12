import pkgutil
from typing import Generator, Iterable, Any
from . import binwrite as binw
from .stencil_db import stencil_database
from collections import defaultdict, deque
from coparun_module import coparun, read_data_mem
import struct
import platform

Operand = type['Net'] | float | int


def get_var_name(var: Any, scope: dict[str, Any] = globals()) -> list[str]:
    return [name for name, value in scope.items() if value is var]


def stencil_db_from_package(arch: str = 'native', optimization: str = 'O3') -> stencil_database:
    if arch == 'native':
        arch = platform.machine()
    stencil_data = pkgutil.get_data(__name__, f"obj/stencils_{arch}_{optimization}.o")
    assert stencil_data, f"stencils_{arch}_{optimization} not found"
    return stencil_database(stencil_data)


generic_sdb = stencil_db_from_package()


class Node:
    def __init__(self) -> None:
        self.args: list[Net] = []
        self.name: str = ''

    def __repr__(self) -> str:
        #return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else self.value})"
        return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else (self.value if isinstance(self, InitVar) else '')})"


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

    def __floordiv__(self, other: Any) -> 'Net':
        return _add_op('floordiv', [self, other])

    def __rfloordiv__(self, other: Any) -> 'Net':
        return _add_op('floordiv', [other, self])

    def __neg__(self) -> 'Net':
        return _add_op('sub', [CPVariable(0), self])

    def __gt__(self, other: Any) -> 'Net':
        return _add_op('gt', [self, other])

    def __lt__(self, other: Any) -> 'Net':
        return _add_op('gt', [other, self])

    def __eq__(self, other: Any) -> 'Net':  # type: ignore
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


class InitVar(Node):
    def __init__(self, value: float | int | bool):
        self.dtype, self.value = _get_data_and_dtype(value)
        self.name = 'const_' + self.dtype
        self.args = []


class Write(Node):
    def __init__(self, net: Net):
        self.name = 'write_' + net.dtype
        self.args = [net]


class Op(Node):
    def __init__(self, typed_op_name: str, args: list[Net]):
        assert not args or any(isinstance(t, Net) for t in args), 'args parameter must be of type list[Net]'
        self.name: str = typed_op_name
        self.args: list[Net] = args


def _add_op(op: str, args: list[Any], commutative: bool = False) -> Net:
    arg_nets = [a if isinstance(a, Net) else CPVariable(a) for a in args]

    if commutative:
        arg_nets = sorted(arg_nets, key=lambda a: a.dtype)

    typed_op = '_'.join([op] + [a.dtype for a in arg_nets])

    if typed_op not in generic_sdb.stencil_definitions:
        raise ValueError(f"Unsupported operand type(s) for {op}: {' and '.join([a.dtype for a in arg_nets])}")

    result_type = generic_sdb.stencil_definitions[typed_op].split('_')[0]

    result_net = Net(result_type, Op(typed_op, arg_nets))

    return result_net


class CPVariable(Net):
    def __init__(self, value: Any):
        vi = InitVar(value)
        Net.__init__(self, vi.dtype, vi)


class CPFloat(CPVariable):
    def __init__(self, value: float):
        super().__init__(value)


class CPInt(CPVariable):
    def __init__(self, value: int):
        super().__init__(value)


class CPBool(CPVariable):
    def __init__(self, value: bool):
        super().__init__(value)


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
    return vec3d((CPVariable(x), CPVariable(y), CPVariable(z)))


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
    return [net_lookup[node] for node in nodes if isinstance(node, InitVar)]


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
        if not isinstance(node, InitVar):
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


def get_variable_mem_layout(variable_list: Iterable[Net], sdb: stencil_database) -> tuple[list[tuple[Net, int, int]], int]:
    offset: int = 0
    object_list: list[tuple[Net, int, int]] = []

    for variable in variable_list:
        lengths = sdb.get_symbol_size('dummy_' + variable.dtype)
        object_list.append((variable, offset, lengths))
        offset += (lengths + 3) // 4 * 4

    return object_list, offset


def get_aux_function_mem_layout(function_names: Iterable[str], sdb: stencil_database) -> tuple[list[tuple[str, int, int]], int]:
    offset: int = 0
    function_list: list[tuple[str, int, int]] = []

    for name in function_names:
        lengths = sdb.get_symbol_size(name)
        function_list.append((name, offset, lengths))
        offset += (lengths + 3) // 4 * 4

    return function_list, offset


def compile_to_instruction_list(node_list: Iterable[Node], sdb: stencil_database) -> tuple[binw.data_writer, dict[Net, tuple[int, int, str]]]:
    variables: dict[Net, tuple[int, int, str]] = dict()

    ordered_ops = list(stable_toposort(get_all_dag_edges(node_list)))
    const_net_list = get_const_nets(ordered_ops)
    output_ops = list(add_read_ops(ordered_ops))
    extended_output_ops = list(add_write_ops(output_ops, const_net_list))

    dw = binw.data_writer(sdb.byteorder)

    # Deallocate old allocated memory (if existing)
    dw.write_com(binw.Command.FREE_MEMORY)

    # Get all nets/variables associated with heap memory
    variable_list = get_nets([[const_net_list]], extended_output_ops)

    # Write data
    variable_mem_layout, data_section_lengths = get_variable_mem_layout(variable_list, sdb)
    dw.write_com(binw.Command.ALLOCATE_DATA)
    dw.write_int(data_section_lengths)

    for net, out_offs, lengths in variable_mem_layout:
        variables[net] = (out_offs, lengths, net.dtype)
        if isinstance(net.source, InitVar):
            dw.write_com(binw.Command.COPY_DATA)
            dw.write_int(out_offs)
            dw.write_int(lengths)
            dw.write_value(net.source.value, lengths)
            # print(f'+ {net.dtype} {net.source.value}')

    # write auxiliary_functions
    aux_function_names = sdb.get_sub_functions(node.name for _, node in extended_output_ops)
    aux_function_mem_layout, aux_function_lengths = get_aux_function_mem_layout(aux_function_names, sdb)
    aux_func_addr_lookup = {name: offs for name, offs, _ in aux_function_mem_layout}

    dw.write_com(binw.Command.COPY_CODE)
    dw.write_int(0)
    dw.write_int(aux_function_lengths)
    for name, _, _ in aux_function_mem_layout:
        dw.write_bytes(sdb.get_function_code(name))

    # Prepare program code and relocations
    object_addr_lookup = {net: offs for net, offs, _ in variable_mem_layout}
    data_list: list[bytes] = []
    patch_list: list[tuple[int, int, int]] = []
    offset = aux_function_lengths  # offset in generated code chunk

    # assemble stencils to main program
    data = sdb.get_function_code('entry_function_shell', 'start')
    data_list.append(data)
    offset += len(data)

    for associated_net, node in extended_output_ops:
        assert node.name in sdb.stencil_definitions, f"- Warning: {node.name} stencil not found"
        data = sdb.get_stencil_code(node.name)
        data_list.append(data)
        # print(f"* {node.name} ({offset}) " + ' '.join(f'{d:02X}' for d in data))

        for patch in sdb.get_patch_positions(node.name):
            if patch.target_symbol_info == 'STT_OBJECT':
                assert associated_net, f"Relocation found but no net defined for operation {node.name}"
                addr = object_addr_lookup[associated_net]
            elif patch.target_symbol_info == 'STT_FUNC':
                addr = aux_func_addr_lookup[patch.target_symbol_name]
            else:
                raise ValueError(f"Unsupported: {node.name} {patch.target_symbol_info} {patch.target_symbol_name}")
            
            patch_value = addr + patch.addend - (offset + patch.addr)
            patch_list.append((patch.type.value, offset + patch.addr, patch_value))
            print('++ ', patch.target_symbol_info, patch.target_symbol_name)

        offset += len(data)

    data = sdb.get_function_code('entry_function_shell', 'end')
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
    for patch_type, patch_addr, addr in patch_list:
        dw.write_com(binw.Command.PATCH_OBJECT)
        dw.write_int(patch_addr)
        dw.write_int(patch_type)
        dw.write_int(addr, signed=True)

    dw.write_com(binw.Command.ENTRY_POINT)
    dw.write_int(aux_function_lengths)

    return dw, variables


class Target():

    def __init__(self, arch: str = 'native', optimization: str = 'O3') -> None:
        self.sdb = stencil_db_from_package(arch, optimization)
        self._variables: dict[Net, tuple[int, int, str]] = dict()

    def compile(self, *variables: Net | list[Net]) -> None:
        nodes: list[Node] = []
        for s in variables:
            if isinstance(s, Net):
                nodes.append(Write(s))
            else:
                for net in s:
                    assert isinstance(net, Net)
                    nodes.append(Write(net))

        dw, self._variables = compile_to_instruction_list(nodes, self.sdb)
        dw.write_com(binw.Command.END_COM)
        assert coparun(dw.get_data()) > 0

    def run(self) -> None:
        # set entry point and run code
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.RUN_PROG)
        dw.write_com(binw.Command.END_COM)
        assert coparun(dw.get_data()) > 0

    def read_value(self, net: Net) -> float | int:
        assert net in self._variables, f"Variable {net} not found"
        addr, lengths, var_type = self._variables[net]
        print('read_value', addr, lengths)
        assert lengths > 0
        data = read_data_mem(addr, lengths)
        assert data is not None and len(data) == lengths, f"Failed to read variable {net}"
        en = {'little': '<', 'big': '>'}[self.sdb.byteorder]
        if var_type == 'float':
            if lengths == 4:
                value = struct.unpack(en + 'f', data)[0]
            elif lengths == 8:
                value = struct.unpack(en + 'd', data)[0]
            else:
                raise ValueError(f"Unsupported float length: {lengths} bytes")
            assert isinstance(value, float)
            return value
        elif var_type == 'int':
            if lengths in (1, 2, 4, 8):
                value = int.from_bytes(data, byteorder=self.sdb.byteorder, signed=True)
                assert isinstance(value, int)
                return value
            else:
                raise ValueError(f"Unsupported int length: {lengths} bytes")
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

    def read_variable_remote(self, net: Net) -> None:
        dw = binw.data_writer(self.sdb.byteorder)
        add_read_command(dw, self._variables, net)
        assert coparun(dw.get_data()) > 0


def add_read_command(dw: binw.data_writer, variables: dict[Net, tuple[int, int, str]], net: Net) -> None:
    assert net in variables, f"Variable {net} not found in data writer variables"
    addr, lengths, _ = variables[net]
    dw.write_com(binw.Command.READ_DATA)
    dw.write_int(addr)
    dw.write_int(lengths)
