from typing import Generator, Iterable, Any, Sequence
from . import _binwrite as binw
from ._stencils import stencil_database, patch_entry, TRICORE_BRANCH_RELOCATIONS
from collections import defaultdict
import heapq
import struct
from ._basic_types import Net, Node, Store, HeadNode, Constant, TimeStep, Op, transl_type, ArrayNet, ArrayConst, ArrayDelay, ArrayOp, ArrayElement, ArrayPack

# Pseudo operation: store register 0 to an array element of an ArrayPack
PACK_STORE = "pack_store"


def is_pack_constant(net: Net) -> bool:
    """Elements of an ArrayPack that are constants are written to the
    array memory when loading the program instead of storing them on each run.
    Inputs (copapy values) can be changed by Target.write_value."""
    return isinstance(net.source, Constant)


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

    # Min-heap of nodes with indegree 0 by first appearance (order values are unique)
    queue = [(order[n], n) for n in indeg if indeg[n] == 0]
    heapq.heapify(queue)
    result: list[Node] = []

    while queue:
        _, node = heapq.heappop(queue)
        result.append(node)

        for nei in adj[node]:
            indeg[nei] -= 1
            if indeg[nei] == 0:
                heapq.heappush(queue, (order[nei], nei))

    # Check if graph had a cycle (not all nodes output)
    if len(result) != len(indeg):
        raise ValueError("Graph contains a cycle — topological sort not possible")

    return result


def _node_key(node: Node, arg_ids: tuple[int, ...]) -> tuple[Any, ...] | None:
    """Returns a Key in form of a tuple for a node that identifies the node in
    the id table of canonicalize: nodes with the same key compute the same value.
    None for nodes that are never merged.

    Arguments:
        node: Node to get the key for
        arg_ids: Ids of the canonical nodes of the arguments
    """
    if isinstance(node, Op):
        if isinstance(node, ArrayOp):
            length = node.result.length if isinstance(node.result, ArrayNet) else None
            return (node.name, arg_ids, node.result.dtype, length)
        return (node.name, tuple(sorted(arg_ids)) if node.commutative else arg_ids)
    if isinstance(node, Constant):
        # Floats by bit pattern: 0.0 and -0.0 are different constants
        bits = struct.pack('d', node.value) if node.dtype == 'float' else node.value
        return ('const', node.dtype, bits)
    if isinstance(node, TimeStep):
        return ('dt', node.factor, node.exponent)
    if not arg_ids:
        return None  # Input and ArrayConst: each node is distinct
    if isinstance(node, ArrayElement):
        return ('element', arg_ids[0], node.index, node.dtype)
    if isinstance(node, Store):
        return (node.name, arg_ids)
    return None  # ArrayPack: each node is distinct


def _modified_node_copy(node: Node, args: tuple[Net, ...]) -> Node:
    """Copy of a node with other argument nets"""
    new = object.__new__(type(node))  # much faster than copy.copy
    new.__dict__.update(node.__dict__)
    new.args = args
    if isinstance(new, ArrayOp):
        res = new.result
        new.result = ArrayNet(res.dtype, new, res.length) if isinstance(res, ArrayNet) else Net(res.dtype, new)
    elif isinstance(new, ArrayPack):
        new.result = ArrayNet(new.result.dtype, new, new.result.length)
        new.elements = [Net(el.dtype, ArrayElement(new.result, i)) for i, el in enumerate(new.elements)]
    return new


def canonicalize(roots: Iterable[Node]) -> tuple[list[Node], dict[Net, Net]]:
    """Rebuild a DAG with all equivalent nodes merged (common subexpression
    elimination). The identity of each node is looked up in an id table.

    The given DAG is not changed: nodes are copied if their arguments are
    replaced, otherwise the nodes and nets are reused. In the returned DAG
    each node has exactly one result net.

    Arguments:
        roots: End nodes of the DAG

    Returns:
        Tuple of the end nodes of the canonical DAG and a dict to look up
        the canonical net for each net of the given DAG
    """
    roots = list(roots)
    table: dict[tuple[Any, ...], int] = {}  # id by key
    ids: dict[Node, int] = {}  # id of each given node, equivalent nodes have the same id
    nodes: list[Node] = []  # canonical node by id
    nets: list[Net | None] = []  # result net of the canonical node by id, None until it is used
    net_map: dict[Net, Net] = {}
    with_result = (ArrayOp, ArrayPack)

    stack = roots[::-1]
    while stack:
        node = stack[-1]
        if node in ids:
            stack.pop()
            continue
        pending = [net.source for net in node.args if net.source not in ids]
        if pending:
            # Arguments first: the node is on top of the stack again when they have ids
            stack.extend(pending)
            continue
        stack.pop()

        new_args: list[Net] = []
        arg_ids: list[int] = []
        changed = False
        for net in node.args:
            i = ids[net.source]
            new = nets[i]
            if new is None:
                new = nets[i] = net if nodes[i] is net.source else Net(net.dtype, nodes[i])
            if new is not net:
                changed = True
            net_map[net] = new
            new_args.append(new)
            arg_ids.append(i)

        key = _node_key(node, tuple(arg_ids))
        i = table.get(key, -1) if key else -1

        if i < 0:
            i = len(nodes)
            canonical = _modified_node_copy(node, tuple(new_args)) if changed else node
            nodes.append(canonical)
            if key:
                table[key] = i
            if isinstance(canonical, with_result):
                nets.append(canonical.result)
                if isinstance(canonical, ArrayPack):
                    # Elements of the array are the nets the values are stored to
                    assert isinstance(node, ArrayPack)
                    for element, old in zip(canonical.elements, node.elements):
                        table[_node_key(element.source, (i,))] = ids[old.source] = len(nodes)  # type: ignore[index]
                        nodes.append(element.source)
                        nets.append(element)
                        net_map[old] = element
            else:
                nets.append(None)

        ids[node] = i
        if isinstance(node, with_result):
            canonical = nodes[i]
            assert isinstance(canonical, with_result)
            net_map[node.result] = canonical.result

    return list(dict.fromkeys(nodes[ids[node]] for node in roots)), net_map


def get_all_dag_edges_between(roots: Iterable[Node], leaves: Iterable[Node]) -> Generator[tuple[Node, Node], None, None]:
    """Get all edges in the DAG connecting given roots with given leaves.
    Leaves not connected to a root yield no edges.

    Arguments:
        nodes: Iterable of nodes to start the traversal from

    Yields:
        Tuples of (source_node, target_node) representing edges in the DAG
    """
    # Walk the full DAG starting from given roots to final leaves
    parent_lookup: dict[Node, set[Node]] = {}
    node_list: list[Node] = list(roots)
    while(node_list):
        node = node_list.pop()
        for net in node.args:
            if net.source in parent_lookup:
                parent_lookup[net.source].add(node)
            else:
                parent_lookup[net.source] = {node}
                node_list.append(net.source)

    # Walk the DAG in reverse direction starting from given leaves to given roots
    emitted_edges: set[tuple[Node, Node]] = set()
    node_list = list(leaves)
    while(node_list):
        child_node = node_list.pop()
        if child_node in parent_lookup:
            for node in parent_lookup[child_node]:
                edge = (child_node, node)
                if edge not in emitted_edges:
                    yield edge
                    node_list.append(node)
                    emitted_edges.add(edge)


def get_all_dag_edges(nodes: Iterable[Node]) -> Generator[tuple[Node, Node], None, None]:
    """Get all edges in the DAG by traversing from the given nodes

    Arguments:
        nodes: Iterable of nodes to start the traversal from

    Yields:
        Tuples of (source_node, target_node) representing edges in the DAG
    """
    emitted_edges: set[tuple[Node, Node]] = set()
    node_list: list[Node] = list(nodes)

    while(node_list):
        node = node_list.pop()
        for net in node.args:
            edge = (net.source, node)
            if edge not in emitted_edges:
                yield edge
                node_list.append(net.source)
                emitted_edges.add(edge)


def get_head_nets(roots: Iterable[Node], node_type: type[Node] | tuple[type[Node], ...]) -> dict[Node, Net]:
    """Head nodes of the given type in the DAG and the net of each

    Arguments:
        roots: End nodes of the DAG
        node_type: Type or types of the head nodes to find

    Returns:
        Dict of the net by head node in the order of discovery
    """
    found: dict[Node, Net] = {}
    visited: set[Node] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for net in node.args:
            if isinstance(net.source, node_type):
                found.setdefault(net.source, net)
            stack.append(net.source)
    return found


def get_const_nets(nodes: list[Node]) -> list[Net]:
    """Get all nets with a constant nodes value

    Returns:
        List of nets whose source node is a Const
    """
    net_lookup = {net.source: net for node in nodes for net in node.args}
    return [net_lookup[node] for node in nodes if isinstance(node, HeadNode)]


def add_load_ops(node_list: list[Node]) -> Generator[tuple[Net | None, Node], None, None]:
    """Add load/read node before each op where arguments are not already positioned
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
        if isinstance(node, ArrayOp):
            # Operands are accessed in memory, register contents are not preserved
            yield None, node
            registers = [None, None]
        elif isinstance(node, ArrayPack):
            # Store each value from register 0 to the memory of its array element,
            # constants are written to the array memory when loading the program
            for net, element in zip(node.args, node.elements):
                if is_pack_constant(net):
                    continue
                if id(net) != id(registers[0]):
                    type_list = ['int' if r is None else transl_type(r.dtype) for r in registers]
                    yield net, Op(f"load_{transl_type(net.dtype)}_reg0_" + '_'.join(type_list), [])
                    registers[0] = net
                yield element, Op(PACK_STORE, [])
        elif not isinstance(node, HeadNode | ArrayConst | ArrayDelay | ArrayElement):
            for i, net in enumerate(node.args):
                if id(net) != id(registers[i]):  # TODO: consider register swap and commutative ops
                    #if net in registers:
                    #    print('x  swap registers')
                    type_list = ['int' if r is None else transl_type(r.dtype) for r in registers]
                    new_node = Op(f"load_{transl_type(net.dtype)}_reg{i}_" + '_'.join(type_list), [])
                    yield net, new_node
                    registers[i] = net

            if node in net_lookup:
                result_net = net_lookup[node]
                yield result_net, node
                registers[0] = result_net
                if len(node.args) < 2:  # Reset virtual register for single argument functions
                    registers[1] = None
            else:
                yield None, node


def add_store_ops(net_node_list: list[tuple[Net | None, Node]], const_nets: list[Net]) -> Generator[tuple[Net | None, Node], None, None]:
    """Add store/write operation for each new defined net if a read operation is later followed

    Returns:
        Yields tuples of a net and a node. The associated net is provided for read and write nodes.
        Otherwise None is returned in the tuple.
    """

    # Initialize set of nets with constants
    stored_nets = set(const_nets)

    #assert all(node.name.startswith('load_') for net, node in net_node_list if net)
    read_back_nets = {
        net for net, node in net_node_list
        if net and node.name.startswith('load_')}

    # Scalar arguments of array ops are read from memory
    read_back_nets |= {
        net for _, node in net_node_list if isinstance(node, ArrayOp)
        for net in node.args if not isinstance(net, ArrayNet)}

    registers: list[Net | None] = [None, None]

    for net, node in net_node_list:
        if isinstance(node, ArrayOp):
            registers = [None, None]

        if node.name == PACK_STORE:
            # Store register 0 to the array element net
            type_list = [transl_type(r.dtype) if r else 'int' for r in registers]
            yield net, Op(f"store_{type_list[0]}_reg0_" + '_'.join(type_list), [])
            continue

        if isinstance(node, Store):
            assert len(registers) == 2
            type_list = [transl_type(r.dtype) if r else 'int' for r in registers]
            yield node.args[0], Op(f"store_{type_list[0]}_reg0_" + '_'.join(type_list), node.args)
        elif node.name.startswith('load_'):
            yield net, node
        else:
            yield None, node

        if net:
            # Update virtual register state with result net and 2. parameter net
            registers[0] = net
            if len(node.args) > 1:
                registers[1] = node.args[1]
            #print("* reg", node.name, [transl_type(r.dtype) if r else 'int' for r in registers])

            if net in read_back_nets and net not in stored_nets and \
               not isinstance(net.source, ArrayOp | ArrayElement):  # these are already in memory
                type_list = [transl_type(r.dtype) if r else 'int' for r in registers]
                yield net, Op(f"store_{type_list[0]}_reg0_" + '_'.join(type_list), [])
                stored_nets.add(net)


def get_nets(*inputs: Iterable[Iterable[Any]]) -> list[Net]:
    """Get all unique nets from the provided inputs
    """
    nets: set[Net] = set()

    for input in inputs:
        for el in input:
            for net in el:
                if isinstance(net, Net):
                    nets.add(net)
                else:
                    assert net is None or isinstance(net, Node), net

    return list(nets)


def get_data_layout(variable_list: Iterable[Net], sdb: stencil_database, offset: int = 0) -> tuple[list[tuple[Net, int, int]], int]:
    """Get memory layout for the provided variables

    Arguments:
        variable_list: Variables to layout
        sdb: Stencil database for size lookup
        offset: Starting offset for layout

    Returns:
        Tuple of list of (variable, start_offset, length) and total length"""

    object_list: list[tuple[Net, int, int]] = []

    for variable in variable_list:
        lengths = sdb.get_type_size(transl_type(variable.dtype))
        alignment = lengths  # align variables to their own size
        if isinstance(variable, ArrayNet):
            lengths *= variable.length
            alignment = 16  # align arrays for SIMD access
        offset = (offset + alignment - 1) // alignment * alignment
        object_list.append((variable, offset, lengths))
        offset += lengths

    return object_list, offset


#def get_target_sym_lookup(function_names: Iterable[str], sdb: stencil_database) -> dict[str, patch_entry]:
#    return {patch.target_symbol_name: patch for name in set(function_names) for patch in sdb.get_patch_positions(name)}


def get_section_layout(section_indexes: Iterable[int], sdb: stencil_database, offset: int = 0) -> tuple[list[tuple[int, int, int]], int]:
    """Get memory layout for the provided sections

    Arguments:
        section_indexes: Sections (by index) to layout
        sdb: Stencil database for size lookup
        offset: Starting offset for layout

    Returns:
        Tuple of list of (section_id, start_offset, length) and total length
    """
    section_list: list[tuple[int, int, int]] = []

    for index in section_indexes:
        lengths = sdb.get_section_size(index)
        alignment = sdb.get_section_alignment(index)
        offset = (offset + alignment - 1) // alignment * alignment
        section_list.append((index, offset, lengths))
        offset += lengths

    return section_list, offset


def get_aux_func_layout(function_names: Iterable[str], sdb: stencil_database, offset: int = 0) -> tuple[list[tuple[int, int, int]], dict[str, int], int]:
    """Get memory layout for the provided auxiliary functions

    Arguments:
        function_names: Function names to layout
        sdb: Stencil database for size lookup
        offset: Starting offset for layout

    Returns:
        Tuple of list of (section_id, start_offset, length), function address lookup dictionary, and total length
    """
    function_lookup: dict[str, int] = {}
    section_list: list[tuple[int, int, int]] = []
    section_cache: dict[int, int] = {}

    for name in function_names:
        index = sdb.get_symbol_section_index(name)

        if index in section_cache:
            section_offset = section_cache[index]
            function_lookup[name] = section_offset + sdb.get_symbol_offset(name)
        else:
            lengths = sdb.get_section_size(index)
            alignment = sdb.get_section_alignment(index)
            offset = (offset + alignment - 1) // alignment * alignment
            section_list.append((index, offset, lengths))
            section_cache[index] = offset
            function_lookup[name] = offset + sdb.get_symbol_offset(name)
            offset += lengths

    return section_list, function_lookup, offset


def get_dag_stats(node_list: Iterable[Node | Net]) -> dict[str, int]:
    """Get operation statistics for the DAG identified by provided end nodes

    Arguments:
        node_list: List of end nodes of the DAG

    Returns:
        Dictionary of operation name to occurrence count
    """
    roots, _ = canonicalize(n.source if isinstance(n, Net) else n for n in node_list)
    edges = get_all_dag_edges(roots)
    ops = {node for node, _ in edges}

    op_stat: dict[str, int] = {}
    for op in ops:
        op_stat[op.name] = op_stat.get(op.name, 0) + 1

    return op_stat


class _Program():
    """Operations and variables of one entry point of a compiled program"""
    def __init__(self, node_list: Iterable[Node], external: dict[Node, int]):
        self.roots, self.net_map = canonicalize(node_list)
        self.ordered_ops = stable_toposort(get_all_dag_edges(self.roots))
        const_net_list = get_const_nets(self.ordered_ops)
        output_ops = list(add_load_ops(self.ordered_ops))
        self.extended_output_ops = list(add_store_ops(output_ops, const_net_list))
        self.external = external

        # Get all nets/variables associated with heap memory
        variable_list = set(get_nets([const_net_list], self.extended_output_ops))

        # Arguments and results of array ops are accessed in memory
        variable_list |= {net for _, node in self.extended_output_ops if isinstance(node, ArrayOp)
                          for net in (*node.args, node.result)}

        # Packed arrays (also if all elements are constants and no store is emitted)
        variable_list |= {node.result for node in self.ordered_ops if isinstance(node, ArrayPack)}

        # Array elements are aliases into the memory of their arrays
        self.element_nets = [net for net in variable_list if isinstance(net.source, ArrayElement)]
        variable_list = {net for net in variable_list if not isinstance(net.source, ArrayElement)}
        variable_list |= {net.source.args[0] for net in self.element_nets}

        # Results of in-place array ops are aliases to the memory of their first argument
        self.inplace_ops = [node for node in self.ordered_ops if isinstance(node, ArrayOp) and node.inplace]
        modified = {node.args[0]: node for node in self.inplace_ops}
        for node in self.ordered_ops:
            for net in node.args:
                assert modified.get(net, node) is node, \
                    f"{modified[net].name} modifies {net}, which is also used by {node.name}"
        variable_list -= {node.result for node in self.inplace_ops}
        self.variable_list = variable_list

        self.stencil_names = {node.name for _, node in self.extended_output_ops}


def compile_programs(programs: Sequence[tuple[Iterable[Node], dict[Node, int]]], sdb: stencil_database,
                     reserved: int = 0) -> tuple[binw.data_writer, list[dict[Net, tuple[int, int, str]]], list[int]]:
    """Compiles several DAGs, each with its own entry point, to one code and
    one data memory. The DAGs share the auxiliary functions and constants.

    Arguments:
        programs: For each entry point the end nodes of its DAG and the addresses
            of external head nodes (Input, ArrayConst): they are not allocated and
            not initialized by the compiler
        sdb: Stencil database
        reserved: Size of a memory block at the start of the data memory that is
            not used by the compiler (e.g. for the external head nodes)

    Returns:
        Tuple of data writer with binary code, variable layout dictionary of
        each program and code offset of each entry point
    """
    data_list: list[bytes] = []  # Binary code of all programs and auxiliary functions
    patch_list: list[patch_entry] = []

    # Prepare programs and get the list of all used stencils
    prepared = [_Program(node_list, external) for node_list, external in programs]

    dw = binw.data_writer(sdb.byteorder)

    # Deallocate old allocated memory (if existing)
    dw.write_com(binw.Command.FREE_MEMORY)

    stencil_names: set[str] = set().union(*(p.stencil_names for p in prepared))
    aux_function_names = sdb.get_sub_functions(stencil_names)
    used_const_sections = sdb.const_sections_from_functions(aux_function_names | stencil_names)

    # Write data
    section_mem_layout, data_length = get_section_layout(used_const_sections, sdb, reserved)
    variable_mem_layouts: list[list[tuple[Net, int, int]]] = []
    for p in prepared:
        layout, data_length = get_data_layout(
            [net for net in p.variable_list if net.source not in p.external], sdb, data_length)
        variable_mem_layouts.append(layout)
    if data_length:
        # Skip for programs without variables or constants
        dw.write_com(binw.Command.ALLOCATE_DATA)
        dw.write_int(data_length)

    # Heap constants
    for section_id, start, lengths in section_mem_layout:
        dw.write_com(binw.Command.COPY_DATA)
        dw.write_int(start)
        dw.write_int(lengths)
        dw.write_bytes(sdb.get_section_data(section_id))

    # prep auxiliary_functions
    code_section_layout, func_addr_lookup, aux_func_len = get_aux_func_layout(aux_function_names, sdb)

    # Prepare program code and relocations
    section_addr_lookup = {id: offs for id, offs, _ in section_mem_layout}

    all_variables: list[dict[Net, tuple[int, int, str]]] = []
    entry_points: list[int] = []
    offset = aux_func_len

    for p, variable_mem_layout in zip(prepared, variable_mem_layouts):
        variables: dict[Net, tuple[int, int, str]] = {}

        # Heap variables
        for net, start, lengths in variable_mem_layout:
            variables[net] = (start, lengths, net.dtype)
            if isinstance(net.source, HeadNode):
                dw.write_com(binw.Command.COPY_DATA)
                dw.write_int(start)
                dw.write_int(lengths)
                dw.write_value(net.source.value, lengths)
                #print(f'+ {net.dtype} {net.source.value}')
            elif isinstance(net.source, ArrayConst | ArrayDelay):
                dw.write_com(binw.Command.COPY_DATA)
                dw.write_int(start)
                dw.write_int(lengths)
                dw.write_bytes(binw.pack_array(net.source.values, net.dtype, sdb.byteorder))
            elif isinstance(net.source, ArrayPack) and any(is_pack_constant(a) for a in net.source.args):
                # Constant elements, the others are stored on each run
                init = [a.source.value if isinstance(a.source, Constant) else 0
                        for a in net.source.args]
                dw.write_com(binw.Command.COPY_DATA)
                dw.write_int(start)
                dw.write_int(lengths)
                dw.write_bytes(binw.pack_array(init, net.dtype, sdb.byteorder))

        # External head nodes
        for net in p.variable_list:
            if net.source in p.external:
                size = sdb.get_type_size(transl_type(net.dtype))
                variables[net] = (p.external[net.source], size * net.length if isinstance(net, ArrayNet) else size, net.dtype)

        # In-place results (in the order of execution, an in-place op can modify an in-place result)
        for op in p.inplace_ops:
            variables[op.result] = variables[op.args[0]]

        object_addr_lookup = {net: offs for net, (offs, _, _) in variables.items()}

        for net in p.element_nets:
            element = net.source
            assert isinstance(element, ArrayElement)
            size = sdb.get_type_size(transl_type(net.dtype))
            object_addr_lookup[net] = object_addr_lookup[element.args[0]] + element.index * size
            variables[net] = (object_addr_lookup[net], size, net.dtype)

        # assemble stencils to main program and patch stencils
        entry_points.append(offset + sdb.thumb_mode)
        data = sdb.get_function_code('entry_function_shell', 'start')
        data_list.append(data)
        #print(f"* entry_function_shell ({offset}) " + ' '.join(f'{d:02X}' for d in data))
        offset += len(data)

        for associated_net, node in p.extended_output_ops:
            assert node.name in sdb.stencil_definitions, f"- Warning: {node.name} stencil not found"
            data = sdb.get_stencil_code(node.name)
            data_list.append(data)
            #print(f"* {node.name} ({offset}) " + ' '.join(f'{d:02X}' for d in data))

            for reloc in sdb.get_relocations(node.name, stencil=True):
                if reloc.target_in_code:
                    # Branch to a label inside the stencil
                    label_addr = offset + reloc.target_symbol_offset - reloc.function_offset - reloc.start
                    patches = sdb.get_patch(reloc, label_addr, offset, binw.PatchFlag.CODE)

                elif reloc.target_symbol_info in ('STT_OBJECT', 'STT_NOTYPE', 'STT_SECTION'):
                    #print('-- ' + reloc.target_symbol_name + ' // ' + node.name)
                    if reloc.target_symbol_name.startswith('ref_'):
                        # Patch addresses of array op arguments (ref_arg<n>) and result (ref_out)
                        assert isinstance(node, ArrayOp), f"Memory reference in non array op {node.name}"
                        slot = reloc.target_symbol_name[4:]
                        ref_net = node.result if slot == 'out' else node.args[int(slot[3:])]
                        patches = sdb.get_patch(reloc, object_addr_lookup[ref_net], offset, binw.PatchFlag.DATA)
                    elif reloc.target_symbol_name.startswith('dummy_'):
                        # Patch for write and read addresses to/from heap variables
                        assert associated_net, f"Relocation found but no net defined for operation {node.name}"
                        #print(f"Patch for write and read addresses to/from heap variables: {node.name} {patch.target_symbol_info} {patch.target_symbol_name}")
                        obj_addr = object_addr_lookup[associated_net]
                        patches = sdb.get_patch(reloc, obj_addr, offset, binw.PatchFlag.DATA)
                    elif reloc.target_symbol_name.startswith('result_'):
                        # Set return jump address to address of following stencil
                        patches = sdb.get_patch(reloc, offset + len(data), offset, binw.PatchFlag.CODE)
                    else:
                        # Patch constants addresses on heap
                        assert reloc.target_section_index in section_addr_lookup, f"- Function or object in {node.name} missing: {reloc.pelfy_reloc.symbol.name}"
                        obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                        patches = sdb.get_patch(reloc, obj_addr, offset, binw.PatchFlag.DATA)
                        #print('* constants stancils', patch.type, patch.patch_address, binw.Command.PATCH_OBJECT, node.name)

                elif reloc.target_symbol_info == 'STT_FUNC':
                    func_addr = func_addr_lookup[reloc.target_symbol_name]
                    patches = sdb.get_patch(reloc, func_addr, offset, binw.PatchFlag.CODE)
                    #print(patch.type, patch.addr, binw.Command.PATCH_FUNC, node.name, '->', patch.target_symbol_name)
                else:
                    raise ValueError(f"Unsupported: {node.name} {reloc.target_symbol_info} {reloc.target_symbol_name}")

                patch_list.extend(patches)

            offset += len(data)

        data = sdb.get_function_code('entry_function_shell', 'end')
        data_list.append(data)
        offset += len(data)

        # Nets of the given DAG are stored at the address of their canonical nets
        for net, canonical in p.net_map.items():
            if canonical in variables:
                variables[net] = variables[canonical]

        all_variables.append(variables)

    # allocate program data
    dw.write_com(binw.Command.ALLOCATE_CODE)
    dw.write_int(offset)

    # write aux functions code
    for i, start, lengths in code_section_layout:
        dw.write_com(binw.Command.COPY_CODE)
        dw.write_int(start)
        dw.write_int(lengths)
        dw.write_bytes(sdb.get_section_data(i))

    # Patch aux functions
    for name, start in func_addr_lookup.items():
        #print('--> ', name, list(sdb.get_relocations(name)))
        for reloc in sdb.get_relocations(name):

            if not reloc.target_section_index:
                if reloc.pelfy_reloc.type in TRICORE_BRANCH_RELOCATIONS:
                    # Branch to a local label without symbol
                    patch_list.extend(sdb.get_patch(reloc, 0, start, binw.PatchFlag.CODE))
                else:
                    assert reloc.pelfy_reloc.type == 'R_ARM_V4BX', (reloc.pelfy_reloc.type, name, reloc.pelfy_reloc.symbol.name)

            elif reloc.target_in_code:
                # Branch to a label inside the function
                label_addr = start + reloc.target_symbol_offset - reloc.function_offset
                patch_list.extend(sdb.get_patch(reloc, label_addr, start, binw.PatchFlag.CODE))

            elif reloc.target_symbol_info in {'STT_OBJECT', 'STT_NOTYPE', 'STT_SECTION'}:
                # Patch constants/variable addresses on heap
                #print('--> DATA ', name, reloc.pelfy_reloc.symbol, reloc.pelfy_reloc.symbol.info, reloc.pelfy_reloc.symbol.section.name)
                assert reloc.target_section_index in section_addr_lookup, f"- Function or object in {name} missing: {reloc.pelfy_reloc.symbol.name}"
                obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                patch_list.extend(sdb.get_patch(reloc, obj_addr, start, binw.PatchFlag.DATA))

            elif reloc.target_symbol_info == 'STT_FUNC':
                #print('--> FUNC', name, reloc.pelfy_reloc.symbol.name, reloc.pelfy_reloc.symbol.info, reloc.pelfy_reloc.symbol.section.name)
                func_addr = func_addr_lookup[reloc.target_symbol_name]
                #print(f'    FUNC {func_addr=}     {start=}')
                patch_list.extend(sdb.get_patch(reloc, func_addr, start, binw.PatchFlag.CODE))

            else:
                raise ValueError(f"Unsupported: {name=} {reloc.target_symbol_info=} {reloc.target_symbol_name=} {reloc.target_section_index}")

    # write entry function code
    dw.write_com(binw.Command.COPY_CODE)
    dw.write_int(aux_func_len)
    dw.write_int(offset - aux_func_len)
    dw.write_bytes(b''.join(data_list))

    # write patch operations
    for patch in patch_list:
        dw.write_com(binw.Command.PATCH)
        dw.write_int(patch.address)
        dw.write_int(patch.value, signed=True)
        dw.write_int(patch.mask)
        dw.write_byte(patch.encoding)
        dw.write_byte(patch.shift)
        dw.write_byte(patch.flags)
        dw.write_byte(0)  # reserved

    return dw, all_variables, entry_points


def compile_to_dag(node_list: Iterable[Node], sdb: stencil_database) -> tuple[binw.data_writer, dict[Net, tuple[int, int, str]]]:
    """Compiles a DAG identified by provided end nodes to binary code

    Arguments:
        node_list: List of end nodes of the DAG to compile
        sdb: Stencil database

    Returns:
        Tuple of data writer with binary code and variable layout dictionary
    """
    dw, (variables,), (entry_point,) = compile_programs([(node_list, {})], sdb)
    dw.write_com(binw.Command.ENTRY_POINT)
    dw.write_int(entry_point)
    return dw, variables
