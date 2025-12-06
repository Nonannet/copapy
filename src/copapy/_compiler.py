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


def get_all_dag_edges_between(roots: Iterable[Node], leaves: Iterable[Node]) -> Generator[tuple[Node, Node], None, None]:
    """Get all edges in the DAG connecting given roots with given leaves

    Arguments:
        nodes: Iterable of nodes to start the traversal from

    Yields:
        Tuples of (source_node, target_node) representing edges in the DAG
    """
    # Walk the full DAG starting from given roots to final leaves
    parent_lookup: dict[Node, set[Node]] = dict()
    node_list: list[Node] = [n for n in roots]
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
    node_list = [n for n in leaves]
    while(node_list):
        child_node = node_list.pop()
        if child_node in parent_lookup:
            for node in parent_lookup[child_node]:
                edge = (child_node, node)
                if edge not in emitted_edges:
                    yield edge
                    node_list.append(node)
                    emitted_edges.add(edge)

    assert all(r in {e[0] for e in emitted_edges} for r in leaves)


def get_all_dag_edges(nodes: Iterable[Node]) -> Generator[tuple[Node, Node], None, None]:
    """Get all edges in the DAG by traversing from the given nodes

    Arguments:
        nodes: Iterable of nodes to start the traversal from

    Yields:
        Tuples of (source_node, target_node) representing edges in the DAG
    """
    emitted_edges: set[tuple[Node, Node]] = set()
    node_list: list[Node] = [n for n in nodes]

    while(node_list):
        node = node_list.pop()
        for net in node.args:
            edge = (net.source, node)
            if edge not in emitted_edges:
                yield edge
                node_list.append(net.source)
                emitted_edges.add(edge)


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
                if id(net) != id(registers[i]):  # TODO: consider register swap and commutative ops
                    #if net in registers:
                    #    print('x  swap registers')
                    type_list = ['int' if r is None else transl_type(r.dtype) for r in registers]
                    new_node = Op(f"read_{transl_type(net.dtype)}_reg{i}_" + '_'.join(type_list), [])
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

    registers: list[Net | None] = [None, None]

    for net, node in net_node_list:
        if isinstance(node, Write):
            assert len(registers) == 2
            type_list = [transl_type(r.dtype) if r else 'int' for r in registers]
            yield node.args[0], Op(f"write_{type_list[0]}_reg0_" + '_'.join(type_list), node.args)
        elif node.name.startswith('read_'):
            yield net, node
        else:
            yield None, node

        if net:
            # Update virtual register state with result net and 2. parameter net
            registers[0] = net
            if len(node.args) > 1:
                registers[1] = node.args[1]
            #print("* reg", node.name, [transl_type(r.dtype) if r else 'int' for r in registers])

            if net in read_back_nets and net not in stored_nets:
                type_list = [transl_type(r.dtype) if r else 'int' for r in registers]
                yield net, Op(f"write_{type_list[0]}_reg0_" + '_'.join(type_list), [])
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
        offset = (offset + lengths - 1) // lengths * lengths  # align variables to their own size
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

    stencil_names = {node.name for _, node in extended_output_ops}
    aux_function_names = sdb.get_sub_functions(stencil_names)
    used_const_sections = sdb.const_sections_from_functions(aux_function_names | stencil_names)

    # Write data
    section_mem_layout, sections_length = get_section_layout(used_const_sections, sdb)
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
    code_section_layout, func_addr_lookup, aux_func_len = get_aux_func_layout(aux_function_names, sdb)

    # Prepare program code and relocations
    object_addr_lookup = {net: offs for net, offs, _ in variable_mem_layout}
    section_addr_lookup = {id: offs for id, offs, _ in section_mem_layout}

    # assemble stencils to main program and patch stencils
    data = sdb.get_function_code('entry_function_shell', 'start')
    data_list.append(data)
    offset = aux_func_len + len(data)

    for associated_net, node in extended_output_ops:
        assert node.name in sdb.stencil_definitions, f"- Warning: {node.name} stencil not found"
        data = sdb.get_stencil_code(node.name)
        data_list.append(data)
        #print(f"* {node.name} ({offset}) " + ' '.join(f'{d:02X}' for d in data))

        for reloc in sdb.get_relocations(node.name, stencil=True):
            if reloc.target_symbol_info in ('STT_OBJECT', 'STT_NOTYPE', 'STT_SECTION'):
                #print('-- ' + reloc.target_symbol_name + ' // ' + node.name)
                if reloc.target_symbol_name.startswith('dummy_'):
                    # Patch for write and read addresses to/from heap variables
                    assert associated_net, f"Relocation found but no net defined for operation {node.name}"
                    #print(f"Patch for write and read addresses to/from heap variables: {node.name} {patch.target_symbol_info} {patch.target_symbol_name}")
                    obj_addr = object_addr_lookup[associated_net]
                    patch = sdb.get_patch(reloc, obj_addr, offset, binw.Command.PATCH_OBJECT.value)
                elif reloc.target_symbol_name.startswith('result_'):
                    # Set return jump address to address of following stencil
                    patch = sdb.get_patch(reloc, offset + len(data), offset, binw.Command.PATCH_FUNC.value)
                else:
                    # Patch constants addresses on heap
                    assert reloc.target_section_index in section_addr_lookup, f"- Function or object in {node.name} missing: {reloc.pelfy_reloc.symbol.name}"
                    obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                    patch = sdb.get_patch(reloc, obj_addr, offset, binw.Command.PATCH_OBJECT.value)
                    #print('* constants stancils', patch.type, patch.patch_address, binw.Command.PATCH_OBJECT, node.name)

            elif reloc.target_symbol_info == 'STT_FUNC':
                func_addr = func_addr_lookup[reloc.target_symbol_name]
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
    for i, start, lengths in code_section_layout:
        dw.write_com(binw.Command.COPY_CODE)
        dw.write_int(start)
        dw.write_int(lengths)
        dw.write_bytes(sdb.get_section_data(i))

    # Patch aux functions
    for name, start in func_addr_lookup.items():
        #print('--> ', name, list(sdb.get_relocations(name)))
        for reloc in sdb.get_relocations(name):

            #assert reloc.target_symbol_info != 'STT_FUNC', "Not tested yet!"

            if not reloc.target_section_index:
                assert reloc.pelfy_reloc.type == 'R_ARM_V4BX'

            elif reloc.target_symbol_info in {'STT_OBJECT', 'STT_NOTYPE', 'STT_SECTION'}:
                # Patch constants/variable addresses on heap
                #print('--> DATA ', name, reloc.pelfy_reloc.symbol, reloc.pelfy_reloc.symbol.info, reloc.pelfy_reloc.symbol.section.name)
                assert reloc.target_section_index in section_addr_lookup, f"- Function or object in {name} missing: {reloc.pelfy_reloc.symbol.name}"
                obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                patch = sdb.get_patch(reloc, obj_addr, start, binw.Command.PATCH_OBJECT.value)
                patch_list.append(patch)

            elif reloc.target_symbol_info == 'STT_FUNC':
                #print('--> FUNC', name, reloc.pelfy_reloc.symbol.name, reloc.pelfy_reloc.symbol.info, reloc.pelfy_reloc.symbol.section.name)
                func_addr = func_addr_lookup[reloc.target_symbol_name]
                patch = sdb.get_patch(reloc, func_addr, start, binw.Command.PATCH_FUNC.value)
                #print(f'    FUNC {func_addr=}     {start=}    {patch.address=}')
                patch_list.append(patch)

            else:
                raise ValueError(f"Unsupported: {name=} {reloc.target_symbol_info=} {reloc.target_symbol_name=} {reloc.target_section_index}")

    # write entry function code
    dw.write_com(binw.Command.COPY_CODE)
    dw.write_int(aux_func_len)
    dw.write_int(offset - aux_func_len)
    dw.write_bytes(b''.join(data_list))

    # write patch operations
    for patch in patch_list:
        dw.write_com(binw.Command(patch.patch_type))
        dw.write_int(patch.address)
        dw.write_int(patch.mask)
        dw.write_int(patch.scale)
        dw.write_int(patch.value, signed=True)

    dw.write_com(binw.Command.ENTRY_POINT)
    dw.write_int(aux_func_len)

    return dw, variables
