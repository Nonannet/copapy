from typing import Generator, Iterable, Any
from . import _binwrite as binw
from ._stencils import stencil_database, patch_entry
from collections import defaultdict, deque
from ._basic_types import Net, Node, Write, CPConstant, Op, transl_type


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
    return [net_lookup[node] for node in nodes if isinstance(node, CPConstant)]


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
        if not isinstance(node, CPConstant):
            for i, net in enumerate(node.args):
                if id(net) != id(registers[i]):
                    #if net in registers:
                    #    print('x  swap registers')
                    type_list = ['int' if r is None else transl_type(r.dtype) for r in registers]
                    new_node = Op(f"read_{transl_type(net.dtype)}_reg{i}_" + '_'.join(type_list), [])
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
    """Get all unique nets from the provided inputs
    """
    nets: set[Net] = set()

    for input in inputs:
        for el in input:
            for net in el:
                if isinstance(net, Net):
                    nets.add(net)

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
        object_list.append((variable, offset, lengths))
        offset += (lengths + 3) // 4 * 4

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
        section_list.append((index, offset, lengths))
        offset += (lengths + 3) // 4 * 4

    return section_list, offset


def get_aux_function_mem_layout(function_names: Iterable[str], sdb: stencil_database, offset: int = 0) -> tuple[list[tuple[str, int, int]], int]:
    """Get memory layout for the provided auxiliary functions

    Arguments:
        function_names: Function names to layout
        sdb: Stencil database for size lookup
        offset: Starting offset for layout

    Returns:
        Tuple of list of (function_name, start_offset, length) and total length
    """
    function_list: list[tuple[str, int, int]] = []

    for name in function_names:
        lengths = sdb.get_symbol_size(name)
        function_list.append((name, offset, lengths))
        offset += (lengths + 3) // 4 * 4

    return function_list, offset


def compile_to_dag(node_list: Iterable[Node], sdb: stencil_database) -> tuple[binw.data_writer, dict[Net, tuple[int, int, str]]]:
    """Compiles a DAG identified by provided end nodes to binary code

    Arguments:
        node_list: List of end nodes of the DAG to compile
        sdb: Stencil database

    Returns:
        Tuple of data writer with binary code and variable layout dictionary
    """
    variables: dict[Net, tuple[int, int, str]] = {}
    data_list: list[bytes] = []
    patch_list: list[patch_entry] = []

    ordered_ops = list(stable_toposort(get_all_dag_edges(node_list)))
    const_net_list = get_const_nets(ordered_ops)
    output_ops = list(add_read_ops(ordered_ops))
    extended_output_ops = list(add_write_ops(output_ops, const_net_list))

    dw = binw.data_writer(sdb.byteorder)

    # Deallocate old allocated memory (if existing)
    dw.write_com(binw.Command.FREE_MEMORY)

    # Get all nets/variables associated with heap memory
    variable_list = get_nets([[const_net_list]], extended_output_ops)

    stencil_names = [node.name for _, node in extended_output_ops]
    aux_function_names = sdb.get_sub_functions(stencil_names)
    used_sections = sdb.const_sections_from_functions(aux_function_names | set(stencil_names))

    # Write data
    section_mem_layout, sections_length = get_section_layout(used_sections, sdb)
    variable_mem_layout, variables_data_lengths = get_data_layout(variable_list, sdb, sections_length)
    dw.write_com(binw.Command.ALLOCATE_DATA)
    dw.write_int(variables_data_lengths)

    # Heap constants
    for section_id, start, lengths in section_mem_layout:
        dw.write_com(binw.Command.COPY_DATA)
        dw.write_int(start)
        dw.write_int(lengths)
        dw.write_bytes(sdb.get_section_data(section_id))

    # Heap variables
    for net, start, lengths in variable_mem_layout:
        variables[net] = (start, lengths, net.dtype)
        if isinstance(net.source, CPConstant):
            dw.write_com(binw.Command.COPY_DATA)
            dw.write_int(start)
            dw.write_int(lengths)
            dw.write_value(net.source.value, lengths)
            #print(f'+ {net.dtype} {net.source.value}')

    # prep auxiliary_functions
    aux_function_mem_layout, aux_function_lengths = get_aux_function_mem_layout(aux_function_names, sdb)
    aux_func_addr_lookup = {name: offs for name, offs, _ in aux_function_mem_layout}

    # Prepare program code and relocations
    object_addr_lookup = {net: offs for net, offs, _ in variable_mem_layout}
    section_addr_lookup = {id: offs for id, offs, _ in section_mem_layout}

    # assemble stencils to main program and patch stencils
    data = sdb.get_function_code('entry_function_shell', 'start')
    data_list.append(data)
    offset = aux_function_lengths + len(data)

    for associated_net, node in extended_output_ops:
        assert node.name in sdb.stencil_definitions, f"- Warning: {node.name} stencil not found"
        data = sdb.get_stencil_code(node.name)
        data_list.append(data)
        #print(f"* {node.name} ({offset}) " + ' '.join(f'{d:02X}' for d in data))

        for reloc in sdb.get_relocations(node.name, stencil=True):
            if reloc.target_symbol_info in {'STT_OBJECT', 'STT_NOTYPE'}:
                #print('-- ' + reloc.target_symbol_name + ' // ' + node.name)
                if reloc.target_symbol_name.startswith('dummy_'):
                    # Patch for write and read addresses to/from heap variables
                    assert associated_net, f"Relocation found but no net defined for operation {node.name}"
                    #print(f"Patch for write and read addresses to/from heap variables: {node.name} {patch.target_symbol_info} {patch.target_symbol_name}")
                    obj_addr = object_addr_lookup[associated_net]
                    patch = sdb.get_patch(reloc, obj_addr, offset, binw.Command.PATCH_OBJECT.value)
                elif reloc.target_symbol_name.startswith('result_'):
                    raise Exception(f"Stencil {node.name} seams to branch to multiple result_* calls.")
                else:
                    # Patch constants addresses on heap
                    obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                    patch = sdb.get_patch(reloc, obj_addr, offset, binw.Command.PATCH_OBJECT.value)
                    #print('* constants stancils', patch.type, patch.patch_address, binw.Command.PATCH_OBJECT, node.name)

            elif reloc.target_symbol_info == 'STT_FUNC':
                func_addr = aux_func_addr_lookup[reloc.target_symbol_name]
                patch = sdb.get_patch(reloc, func_addr, offset, binw.Command.PATCH_FUNC.value)
                #print(patch.type, patch.addr, binw.Command.PATCH_FUNC, node.name, '->', patch.target_symbol_name)
            else:
                raise ValueError(f"Unsupported: {node.name} {reloc.target_symbol_info} {reloc.target_symbol_name}")

            patch_list.append(patch)

        offset += len(data)

    data = sdb.get_function_code('entry_function_shell', 'end')
    data_list.append(data)
    offset += len(data)

    # allocate program data
    dw.write_com(binw.Command.ALLOCATE_CODE)
    dw.write_int(offset)

    # write aux functions code
    for name, start, lengths in aux_function_mem_layout:
        dw.write_com(binw.Command.COPY_CODE)
        dw.write_int(start)
        dw.write_int(lengths)
        dw.write_bytes(sdb.get_function_code(name))

    # Patch aux functions
    for name, start, _ in aux_function_mem_layout:
        for reloc in sdb.get_relocations(name):

            assert reloc.target_symbol_info != 'STT_FUNC', "Not tested yet!"

            if reloc.target_symbol_info in {'STT_OBJECT', 'STT_NOTYPE', 'STT_SECTION'}:
                # Patch constants/variable addresses on heap
                obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                patch = sdb.get_patch(reloc, obj_addr, start, binw.Command.PATCH_OBJECT.value)
                #print('* constants aux', patch.type, patch.patch_address, obj_addr, binw.Command.PATCH_OBJECT, name)

            elif reloc.target_symbol_info == 'STT_FUNC':
                func_addr = aux_func_addr_lookup[reloc.target_symbol_name]
                patch = sdb.get_patch(reloc, func_addr, offset, binw.Command.PATCH_FUNC.value)

            else:
                raise ValueError(f"Unsupported: {name=} {reloc.target_symbol_info=} {reloc.target_symbol_name=} {reloc.target_section_index}")

            patch_list.append(patch)

    #assert False, aux_function_mem_layout

    # write entry function code
    dw.write_com(binw.Command.COPY_CODE)
    dw.write_int(aux_function_lengths)
    dw.write_int(offset - aux_function_lengths)
    dw.write_bytes(b''.join(data_list))

    # write patch operations
    for patch in patch_list:
        dw.write_com(binw.Command(patch.patch_type))
        dw.write_int(patch.address)
        dw.write_int(patch.mask)
        dw.write_int(patch.scale)
        dw.write_int(patch.value, signed=True)

    dw.write_com(binw.Command.ENTRY_POINT)
    dw.write_int(aux_function_lengths)

    return dw, variables
