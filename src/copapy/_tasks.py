"""Tasks: parts of a copapy program that run periodically or on request on a
target. The tasks of a target share one code and one data memory section.

Tasks follow the Logical Execution Time model: a task reads its inputs (copapy
values written by the host and outputs of other tasks) at its release and its
outputs become visible at the end of its period, i.e. at its next release.
The results do not depend on execution times or the order of execution.
"""
import math
import struct
from typing import Any, Iterable
from . import _binwrite as binw
from ._basic_types import Net, Node, Store, Input, HeadNode, Delay, ArrayDelay, TimeStep, ArrayNet, ArrayConst, ArrayOp, ArrayPack, ArrayElement, value, transl_type
from ._arrays import array, ArrayType
from ._compiler import compile_programs, get_head_nets, _modified_node_copy
from ._target import _TargetBase, _LocalMemory, collect_roots
from ._io import IoChannel, IoChannels

# task_t, copy_t and io_t records in tasks.h
_TASK_FORMAT = 'IIiIIIIIIIIIIIIIIIIIIIIII'
_TASK_SIZE = struct.calcsize('<' + _TASK_FORMAT)
_COPY_FORMAT = 'III'
_COPY_SIZE = struct.calcsize('<' + _COPY_FORMAT)
_IO_FORMAT = 'II'
_IO_SIZE = struct.calcsize('<' + _IO_FORMAT)


class Task():
    """Part of a copapy program that computes the given outputs.

    A periodic task is released every period, starting at time 0. A task
    without period only runs on request by exec of the target.

    A task reads its inputs at its release and publishes its outputs at the
    end of its period: copapy values (inputs written by the host) and outputs
    of other tasks are read from the copies made at the release. Everything
    else required for the outputs is computed by the task itself.
    """
    def __init__(self, outputs: Any, period: float | None = None, priority: int | None = None, mem: Any = None) -> None:
        """Create a task.

        Arguments:
            outputs: Values, vectors, tensors and arrays computed by the task, also
                in (nested) sequences or as values of dicts. They can be read by
                read_value of the target and used by other tasks.
            period: Period in seconds, a multiple of the tick of the target. None
                for a task that only runs on request (exec).
            priority: Tasks with higher priority preempt tasks with lower priority.
                None is priority 0. Of tasks with equal priority the task with the
                shorter period runs first.
            mem: Memory region for the data of the task on targets with several
                memory regions
        """
        if period is not None and not period > 0:
            raise ValueError(f"Period must be positive, not {period}")
        self.outputs = outputs
        self.period = period
        self.priority = priority
        self.mem = mem
        self._roots: list[Node] = []
        self._output_nets: list[Net] = []
        for node in dict.fromkeys(collect_roots(outputs)):
            assert isinstance(node, Store | ArrayOp | ArrayPack)
            net = node.args[0] if isinstance(node, Store) else node.result
            if isinstance(net.source, HeadNode) and not isinstance(net.source, Delay):
                continue  # Inputs and constants are not computed
            self._roots.append(node)
            self._output_nets.append(net)

    def __repr__(self) -> str:
        return f"Task(period={self.period}, priority={self.priority})"


def _isolate(roots: list[Node], foreign: set[Net]) -> tuple[list[Node], dict[Net, Net], dict[Input | ArrayConst, list[Net]], dict[Net, Net]]:
    """Copy of a DAG in which the foreign nets (outputs of other tasks) are
    replaced by new head nodes. The DAG is not changed, nodes are only copied
    if their arguments are replaced.

    Arguments:
        roots: End nodes of the DAG
        foreign: Nets to replace

    Returns:
        Tuple of the end nodes of the copy, the new head net by foreign net,
        the nets of each Input and ArrayConst node and a dict to look up the
        net in the copy by the net of the DAG
    """
    new_nodes: dict[Node, Node] = {}
    result_nets: dict[Node, Net] = {}
    net_map: dict[Net, Net] = {}
    cuts: dict[Net, Net] = {}
    heads: dict[Input | ArrayConst, list[Net]] = {}

    def mapped(net: Net) -> Net:
        if net in net_map:
            return net_map[net]
        new: Net
        if net in foreign:
            if isinstance(net, ArrayNet):
                new = ArrayNet(net.dtype, ArrayConst([0] * net.length, net.dtype), net.length)
            else:
                new = Net(net.dtype, Input(0.0 if net.dtype == 'float' else 0))
            cuts[net] = new
        else:
            source = new_nodes[net.source]
            if source is net.source:
                new = net
            elif isinstance(source, ArrayOp | ArrayPack) and net is getattr(net.source, 'result', None):
                new = source.result
            elif source in result_nets:
                new = result_nets[source]
            else:
                new = result_nets[source] = Net(net.dtype, source)
        net_map[net] = new
        return new

    stack = roots[::-1]
    while stack:
        node = stack[-1]
        if node in new_nodes:
            stack.pop()
            continue
        pending = [net.source for net in node.args if net not in foreign and net.source not in new_nodes]
        if pending:
            # Arguments first: the node is on top of the stack again when they are copied
            stack.extend(pending)
            continue
        stack.pop()
        for net in node.args:
            if net not in foreign and isinstance(net.source, Input | ArrayConst):
                nets = heads.setdefault(net.source, [])
                if net not in nets:
                    nets.append(net)
        args = tuple(mapped(net) for net in node.args)
        changed = any(a is not b for a, b in zip(args, node.args))
        new_nodes[node] = _modified_node_copy(node, args) if changed else node

    for node in roots:
        if isinstance(node, ArrayOp | ArrayPack):
            mapped(node.result)

    return [new_nodes[node] for node in roots], cuts, heads, net_map


def _isolate_task(task: Task, foreign: set[Net]) -> tuple[tuple[list[Node], dict[Net, Net], dict[Input | ArrayConst, list[Net]], dict[Net, Net]], dict[Node, Net]]:
    """Isolated DAG of a task (see _isolate) including the computation of the
    nets delayed by the delays of the task, they are stored for the update
    of the delays after each run.

    Returns:
        Tuple of the result of _isolate and the net of each delay of the task
    """
    roots = list(task._roots)
    stored: set[Node] = set()
    while True:
        isolated = _isolate(roots, foreign)
        delays = get_head_nets(isolated[0], (Delay, ArrayDelay))
        new = [d for d in delays if d not in stored]
        if not new:
            return isolated, delays
        for node in new:
            assert isinstance(node, Delay | ArrayDelay)
            if node.input is None:
                raise ValueError(f"The feedback of {delays[node]} is not closed")
            stored.add(node)
            if isinstance(node, ArrayDelay):
                # Arrays are computed in memory
                assert isinstance(node.input.source, ArrayOp | ArrayPack)
                roots.append(node.input.source)
            else:
                roots.append(Store(node.input))


def _sort_state_copies(copies: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Orders the copies updating the delays: a delay that is the input of
    another delay is read before it is overwritten

    Arguments:
        copies: (destination, source, size) of each copy, each destination is
            the memory of one delay
    """
    pending = [c for c in copies if c[0] != c[1]]
    ordered: list[tuple[int, int, int]] = []
    while pending:
        ready = [c for c in pending if not any(o is not c and o[1] == c[0] for o in pending)]
        if not ready:
            raise ValueError("Delays forming a cycle without any computation are not supported")
        ordered.extend(ready)
        pending = [c for c in pending if all(c is not r for r in ready)]
    return ordered


def _task_list(args: Iterable[Any]) -> list[Task]:
    """Tasks from the arguments of deploy: values that are no tasks form one task"""
    tasks: list[Task] = []
    values: list[Any] = []
    for arg in args:
        if isinstance(arg, Task):
            tasks.append(arg)
        elif isinstance(arg, Iterable) and not isinstance(arg, value | array | ArrayType | dict | str | bytes):
            items = list(arg)
            if items and all(isinstance(t, Task) for t in items):
                tasks.extend(items)
            elif any(isinstance(t, Task) for t in items):
                raise TypeError("A sequence must contain either tasks or values")
            else:
                values.append(items)
        else:
            values.append(arg)
    if values:
        tasks.append(Task(values))
    if not tasks:
        raise ValueError("Nothing to deploy")
    return list(dict.fromkeys(tasks))


def _pack_head(node: Input | ArrayConst, size: int, byteorder: binw.ByteOrder) -> bytes:
    """Initial memory content of an input"""
    if isinstance(node, ArrayConst):
        return binw.pack_array(node.values, node.dtype, byteorder)
    if isinstance(node.value, float):
        return struct.pack({'little': '<', 'big': '>'}[byteorder] + ('f' if size == 4 else 'd'), node.value)
    return node.value.to_bytes(size, byteorder, signed=True)


class _TaskTarget(_TargetBase):
    """Deployment of tasks on a target and access to their inputs and outputs"""
    tick: float

    def _init_tasks(self, tick: float) -> None:
        self.tick = tick
        self._tasks: list[Task] = []
        self._task_table = 0  # Offset of the task table in the data memory
        self._ticks = 0  # Simulated time
        self.io = IoChannels([])  # I/O channels of the device
        self._bindings: dict[tuple[IoChannel, bool], Net] = {}  # Bound value by I/O channel and reading

    def _period_ticks(self, task: Task) -> int:
        if task.period is None:
            return 0
        ticks = round(task.period / self.tick)
        if ticks < 1 or abs(ticks * self.tick - task.period) > 1e-6 * self.tick:
            raise ValueError(f"Period {task.period} of {task} is no multiple of the tick {self.tick} s")
        return ticks

    def deploy(self, *tasks: Any) -> None:
        """Compiles the tasks for the target and loads them. Previously
        deployed tasks are removed.

        Arguments:
            tasks: Tasks or sequences of tasks. All other arguments (values,
                vectors, tensors, arrays and sequences of them) are the outputs
                of one additional task without period.
        """
        task_list = sorted(_task_list(tasks), key=lambda t: (-(t.priority or 0), t.period or math.inf))
        if len(task_list) > 32:
            raise ValueError(f"At most 32 tasks are supported, not {len(task_list)}")
        periods = [self._period_ticks(t) for t in task_list]

        owner: dict[Net, Task] = {}  # Net to the task that computes it
        for task in task_list:
            for net in task._output_nets:
                if owner.setdefault(net, task) is not task:
                    raise ValueError(f"{net} is an output of {owner[net]} and of {task}")

        isolated: list[tuple[list[Node], dict[Net, Net], dict[Input | ArrayConst, list[Net]], dict[Net, Net]]] = []
        delays: list[dict[Node, Net]] = []  # Net of each delay by task
        delay_owner: dict[Node, Task] = {}
        for task in task_list:
            iso, task_delays = _isolate_task(task, {net for net, t in owner.items() if t is not task})
            for node, net in task_delays.items():
                if delay_owner.setdefault(node, task) is not task:
                    raise ValueError(f"The delay {net} is used by {delay_owner[node]} and by {task}, "
                                     "use it as output of one task in the other task")
            isolated.append(iso)
            delays.append(task_delays)

        sdb = self.sdb
        offset = 0

        def alloc(net: Net) -> tuple[int, int]:
            nonlocal offset
            size = sdb.get_type_size(transl_type(net.dtype))
            align = 16 if isinstance(net, ArrayNet) else size
            if isinstance(net, ArrayNet):
                size *= net.length
            offset = (offset + align - 1) // align * align
            offset += size
            return offset - size, size

        values: dict[Net, tuple[int, int, str]] = {}  # Dict of the address, size and type of each net
        init: list[tuple[int, bytes]] = []  # Initial memory content of the inputs and constants

        # Task table
        offset = _TASK_SIZE * len(task_list)
        task_table = 0

        # Memory of the inputs, written by the host
        input_addr: dict[Input | ArrayConst, tuple[int, int]] = {}
        for _, _, heads, _ in isolated:
            for node, nets in heads.items():
                if node not in input_addr:
                    input_addr[node] = alloc(nets[0])
                    init.append((input_addr[node][0], _pack_head(node, input_addr[node][1], sdb.byteorder)))
                for net in nets:
                    values[net] = (*input_addr[node], net.dtype)

        # Published outputs
        for net in owner:
            values[net] = (*alloc(net), net.dtype)

        # Copies of the inputs and outputs of other tasks in the memory of each task
        externals: list[dict[Node, int]] = []
        latch_copies: list[list[tuple[int, int, int]]] = []
        for task, (roots, cuts, heads, _) in zip(task_list, isolated):
            external: dict[Node, int] = {}
            copies: list[tuple[int, int, int]] = []
            time_steps: dict[tuple[float, int], int] = {}  # Address of each time constant of the task
            for node, net in get_head_nets(roots, TimeStep).items():
                if task.period is None:
                    raise ValueError(f"{task} uses dt but has no period, "
                                     "a time step must be given explicitly (dt argument)")
                assert isinstance(node, TimeStep)
                key = (node.factor, node.exponent)
                if key not in time_steps:
                    addr, size = alloc(net)
                    time_steps[key] = addr
                    constant = node.factor * task.period ** node.exponent
                    init.append((addr, struct.pack({'little': '<', 'big': '>'}[sdb.byteorder] + ('f' if size == 4 else 'd'), constant)))
                external[node] = time_steps[key]
            for node, nets in heads.items():
                addr, size = input_addr[node]
                if isinstance(node, ArrayConst):
                    external[node] = addr  # Arrays are read in place
                else:
                    external[node] = alloc(nets[0])[0]
                    copies.append((external[node], addr, size))
            for net, new in cuts.items():
                addr, size, _ = values[net]
                external[new.source] = alloc(new)[0]
                copies.append((external[new.source], addr, size))
            externals.append(external)
            latch_copies.append(copies)

        # Device I/O: input channels are read into the memory of the input at
        # the release of each task using it, output channels are written from
        # the published output of the task computing it
        io_in: list[list[tuple[int, int]]] = [[] for _ in task_list]
        io_out: list[list[tuple[int, int]]] = [[] for _ in task_list]
        for (channel, reads), net in self._bindings.items():
            if reads:
                users = [i for i, (_, _, heads, _) in enumerate(isolated) if net.source in heads]
                if not users:
                    raise ValueError(f"{net} bound to {channel} is not used by a deployed task")
                for i in users:
                    io_in[i].append((channel.index, input_addr[net.source][0]))  # type: ignore[index]
            elif net not in owner:
                raise ValueError(f"{net} bound to {channel} is no output of a deployed task")
            else:
                io_out[task_list.index(owner[net])].append((channel.index, values[net][0]))

        # Copy and I/O records, the publish records are filled after compiling
        copy_offs: list[tuple[int, int]] = []
        io_offs: list[tuple[int, int]] = []
        for i, (task, copies) in enumerate(zip(task_list, latch_copies)):
            offset = (offset + 3) // 4 * 4
            copy_offs.append((offset, offset + _COPY_SIZE * len(copies)))
            offset += _COPY_SIZE * (len(copies) + len(task._output_nets) + len(delays[i]))
            io_offs.append((offset, offset + _IO_SIZE * len(io_in[i])))
            offset += _IO_SIZE * (len(io_in[i]) + len(io_out[i]))

        self._release_memory()
        dw, variables, entry_points = compile_programs(
            [(roots, external) for (roots, _, _, _), external in zip(isolated, externals)], sdb, offset)

        en = {'little': '<', 'big': '>'}[sdb.byteorder]
        reserved = bytearray(offset)
        for addr, data in init:
            reserved[addr:addr + len(data)] = data
        for i, task in enumerate(task_list):
            net_map = isolated[i][3]
            publish = [(values[net][0], *variables[i][net_map[net]][:2]) for net in task._output_nets]
            # Delays are updated after publishing (a delay can be an output)
            publish += _sort_state_copies(
                [(variables[i][net_map[net]][0], *variables[i][net_map[node.input]][:2])  # type: ignore[attr-defined]
                 for node, net in delays[i].items()])
            latch_offs, publish_offs = copy_offs[i]
            for j, copy in enumerate(latch_copies[i] + publish):
                struct.pack_into(en + _COPY_FORMAT, reserved, latch_offs + j * _COPY_SIZE, *copy)
            io_in_offs, io_out_offs = io_offs[i]
            for j, io in enumerate(io_in[i] + io_out[i]):
                struct.pack_into(en + _IO_FORMAT, reserved, io_in_offs + j * _IO_SIZE, *io)
            struct.pack_into(en + _TASK_FORMAT, reserved, task_table + i * _TASK_SIZE,
                             entry_points[i], periods[i], task.priority or 0,
                             latch_offs, len(latch_copies[i]), publish_offs, len(publish),
                             *[0] * 14,
                             io_in_offs, len(io_in[i]), io_out_offs, len(io_out[i]))

        dw.write_com(binw.Command.COPY_DATA)
        dw.write_int(0)
        dw.write_int(len(reserved))
        dw.write_bytes(bytes(reserved))
        dw.write_com(binw.Command.ENTRY_POINT)
        dw.write_int(entry_points[0])
        dw.write_com(binw.Command.TASK_TABLE)
        dw.write_int(task_table)
        dw.write_int(len(task_list))
        self._send(dw)

        self._values = values
        self._tasks = task_list
        self._task_table = task_table
        self._ticks = 0

    def _task_index(self, task: Task | None) -> int:
        if task is None:
            if len(self._tasks) != 1:
                raise ValueError(f"{len(self._tasks)} tasks deployed, the task to run must be specified")
            return 0
        if task not in self._tasks:
            raise ValueError(f"{task} is not deployed on the target")
        return self._tasks.index(task)

    def exec(self, task: Task | None = None) -> None:
        """Runs one job of a deployed task: reading inputs, computing and
        publishing the outputs.

        Arguments:
            task: Task to run, can be omitted if only one task is deployed
        """
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.RUN_TASK)
        dw.write_int(self._task_index(task))
        self._send(dw)

    def stats(self, task: Task | None = None) -> dict[str, Any]:
        """Statistics of a deployed task. On devices the statistics are reset
        by run().

        Arguments:
            task: Task, can be omitted if only one task is deployed

        Returns:
            Dict with the keys:

            - ``runs``: Number of jobs run
            - ``overruns``: Number of skipped releases because the previous
              job was not finished
            - ``exec_time``: CPU time of the last job in seconds (without
              the time the job was preempted by other tasks)
            - ``exec_time_max``: Maximum CPU time of a job in seconds
            - ``exec_time_mean``: Mean CPU time of the jobs in seconds
            - ``response_time_max``: Maximum time from the release to the end
              of a job in seconds (must be below the period)
            - ``start_delay_max``: Maximum time from the release to the start
              of a job in seconds
            - ``degraded``: The task currently runs with lower priority than
              the communication with the host, because the tasks used
              too much CPU time (load shedding on devices)
            - ``degradations``: Number of times the task was degraded
        """
        addr = self._task_table + self._task_index(task) * _TASK_SIZE
        fields = struct.unpack({'little': '<', 'big': '>'}[self.sdb.byteorder] + _TASK_FORMAT,
                               self._read_mem(addr, _TASK_SIZE))
        runs, overruns = fields[9:11]
        exec_last, exec_max, sum_lo, sum_hi, response_max, delay_max = fields[13:19]
        return {'runs': runs,
                'overruns': overruns,
                'exec_time': exec_last * 1e-9,
                'exec_time_max': exec_max * 1e-9,
                'exec_time_mean': (sum_hi << 32 | sum_lo) * 1e-9 / runs if runs else 0.0,
                'response_time_max': response_max * 1e-9,
                'start_delay_max': delay_max * 1e-9,
                'degraded': bool(fields[19]),
                'degradations': fields[20]}

    def _check_writable(self, net: Net) -> None:
        source = net.source.args[0].source if isinstance(net.source, ArrayElement) else net.source
        if not isinstance(source, Input | ArrayConst):
            raise ValueError(f"{net} is no input, only copapy values and arrays can be written")
        for (channel, reads), bound in self._bindings.items():
            if reads and bound is net:
                raise ValueError(f"{net} is read from the channel {channel.name}")

    def bind(self, variable: value[Any], channel: IoChannel) -> None:
        """Binds a value to an I/O channel of the device. Bindings are used by
        the next deploy.

        A copapy value (input) reads the channel at the release of each task
        using it, it can not be written by write_value anymore. On an output
        channel it reads back the level of the output.

        A value computed by a task writes the output channel when the task
        publishes its outputs. The value must be an output of the task.

        Arguments:
            variable: Value to bind, of the type of the channel (see IoChannel.kind)
            channel: Channel of the target, e.g. target.io.gpio('led')
        """
        if not isinstance(variable, value):
            raise TypeError(f"Only copapy values can be bound to channels, not {type(variable).__name__}")
        if not len(self.io):
            raise ValueError(f"{type(self).__name__} has no I/O channels")
        if channel not in self.io:
            raise ValueError(f"{channel} is no channel of this target, channels: {self.io}")
        if variable.net.dtype != channel.dtype:
            raise TypeError(f"{channel} takes {channel.dtype} values, not {variable.dtype}")
        reads = isinstance(variable.net.source, Input)
        if reads and not channel.readable:
            raise ValueError(f"{channel} can not be read")
        if not reads and channel.input:
            raise ValueError(f"{channel} is an input, only copapy values (inputs) can be bound to it")
        if (channel, reads) in self._bindings:
            raise ValueError(f"{channel} is already bound for {'reading' if reads else 'writing'}")
        if reads and any(r and net is variable.net for (_, r), net in self._bindings.items()):
            raise ValueError(f"{variable} is already bound to a channel")
        self._bindings[(channel, reads)] = variable.net

    @property
    def time(self) -> float:
        """Simulated time in seconds since deploy"""
        return self._ticks * self.tick

    def step(self, until: float) -> None:
        """Runs the periodic tasks in simulated time up to the given time.
        All tasks released before this time are run. Outputs are published up
        to this time: inputs written afterwards are read by the tasks released
        at this time or later.

        Arguments:
            until: Simulated time in seconds since deploy, a multiple of the tick
        """
        ticks = round(until / self.tick)
        if abs(ticks * self.tick - until) > 1e-6 * self.tick:
            raise ValueError(f"Time {until} is no multiple of the tick {self.tick} s")
        if ticks < self._ticks:
            raise ValueError(f"Time {until} is before the current time {self.time}")
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.STEP_TASKS)
        dw.write_int(ticks - self._ticks)
        self._send(dw)
        self._ticks = ticks


class LocalTarget(_LocalMemory, _TaskTarget):
    """Target running tasks in this process with simulated time"""
    def __init__(self, arch: str = 'native', optimization: str = 'O3', tick: float = 1e-4) -> None:
        """Initialize the target

        Arguments:
            arch: Target architecture
            optimization: Optimization level
            tick: Time step of the simulation in seconds, the periods of the tasks
                must be multiples of it
        """
        _LocalMemory.__init__(self, arch, optimization)
        self._init_tasks(tick)
