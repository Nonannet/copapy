"""State machines of periodic tasks. The states and the transitions between
them are declared in a subclass of StateMachine, the conditions of the
transitions are given at instantiation:

    class Pump(cp.StateMachine):
        RUN = cp.State(initial=True)
        STANDBY = cp.State()
        SHUTDOWN = cp.State()

        trip = cp.ANY >> SHUTDOWN  # Declaration order is priority
        idle = RUN >> STANDBY
        start = STANDBY >> RUN
        recover = SHUTDOWN >> STANDBY

    pump = Pump(trip=pressure > 12.0, idle=cp.after(speed < 50.0, 300.0),
                start=start_cmd, recover=ack & (pressure < 10.0))
    motor = pump.select({Pump.RUN: pid_out}, default=0.0)

The state is a signal like any other value. At each run of the task at most
one transition fires: the first declared one whose source is the state of the
previous run and whose condition is true. The new state applies to the same
run, so outputs selected by the state react without delay.
"""
import itertools
from typing import Any, ClassVar, Mapping, overload
from ._basic_types import value, unifloat, uniint, NumLike, to_float, value_from_number
from ._temporal import delay, feedback, dt
from ._flow import iif, TContainer
from ._logic import truth, as_bool, runs_limit

# Placeholders of states have unique numbers until their class is created
_placeholder_numbers = itertools.count(-2, -1)

# Run count of time_in: saturates instead of overflowing
_MAX_RUNS = 2 ** 31 - 2


class State(int):
    """State of a StateMachine, declared as class attribute. It is the
    number of the state, so pump.state == Pump.RUN is a boolean value.

    Transitions are declared by >> from a state or a set of states
    (Pump.RUN | Pump.STANDBY) or from ANY other state.
    """
    name: str
    machine: str
    initial: bool

    def __new__(cls, initial: bool = False) -> 'State':
        return cls._create(next(_placeholder_numbers), initial)

    @classmethod
    def _create(cls, number: int, initial: bool, name: str = '', machine: str = '') -> 'State':
        self = super().__new__(cls, number)
        self.initial = initial
        self.name = name
        self.machine = machine
        return self

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"{self.machine}.{self.name}" if self.machine else f"State({self.name})"

    def __rshift__(self, target: 'State') -> 'Transition':  # type: ignore[override]
        return Transition((self,), target)

    def __or__(self, other: 'State | Sources') -> 'Sources':  # type: ignore[override]
        return Sources((self,)) | other


class Sources:
    """Source states of a transition: a set of states or ANY state"""
    def __init__(self, states: 'tuple[State, ...] | None'):
        self.states = states

    def __or__(self, other: 'State | Sources') -> 'Sources':
        others = (other,) if isinstance(other, State) else other.states
        if self.states is None or others is None:
            return ANY
        return Sources(self.states + tuple(s for s in others if all(s is not t for t in self.states)))

    def __rshift__(self, target: State) -> 'Transition':
        return Transition(self.states, target)

    def __repr__(self) -> str:
        return 'ANY' if self.states is None else ' | '.join(repr(s) for s in self.states)


ANY = Sources(None)
"""Source of a transition from any state except its target"""


class Transition:
    """Transition between states, declared as class attribute of a
    StateMachine. Its name is the keyword for its condition."""
    def __init__(self, sources: 'tuple[State, ...] | None', target: State):
        self.sources = sources
        self.target = target
        self.name = ''

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name}: {Sources(self.sources)} >> {self.target!r}"


class after:
    """Condition of a transition that is true continuously since at least
    seconds while the state machine is in the source state of the transition.
    The time restarts when the source state is entered, the run entering it
    counts.

        >>> pump = Pump(idle=cp.after(speed < 50.0, 300.0), ...)

    Outside of a state machine held is the equivalent.
    """
    def __init__(self, condition: NumLike, seconds: unifloat, dt: 'unifloat | None' = None):
        self.condition = condition
        self.seconds = seconds
        self.dt = dt


class StateMachine:
    """Base class of state machines (see module documentation).

    Attributes:
        states: The states in declaration order, indexed by their number:
            Pump.states[tg.read_value(pump.state)] is the state on the host
        transitions: The transitions in declaration order (priority)
        initial: The state at the first run: the one declared with
            initial=True or the first one
        state: The state as int value
    """
    states: ClassVar[tuple[State, ...]] = ()
    transitions: ClassVar[tuple[Transition, ...]] = ()
    initial: ClassVar[State]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        attributes = list(vars(cls).items())
        placeholders = [s for _, s in attributes if isinstance(s, State)]
        if not placeholders:
            return  # Abstract base of state machines
        initials = [s for s in placeholders if s.initial]
        if len(initials) > 1:
            raise TypeError(f"{cls.__name__} has more than one initial state: {initials}")
        first = initials[0] if initials else placeholders[0]

        states: dict[int, State] = {}  # By id of the placeholder
        for number, p in enumerate(placeholders):
            states[id(p)] = State._create(number, p is first, p.name, cls.__name__)
            setattr(cls, p.name, states[id(p)])

        def state(s: State, transition: str) -> State:
            if id(s) not in states:
                raise TypeError(f"Transition {transition} of {cls.__name__} uses {s!r} of another state machine")
            return states[id(s)]

        transitions: list[Transition] = []
        for name, t in attributes:
            if isinstance(t, Sources):
                raise TypeError(f"{cls.__name__}.{name} has no target state, use >> to declare a transition")
            if isinstance(t, Transition):
                sources = None if t.sources is None else tuple(state(s, name) for s in t.sources)
                transition = Transition(sources, state(t.target, name))
                transition.name = name
                setattr(cls, name, transition)
                transitions.append(transition)

        cls.states = tuple(states.values())
        cls.transitions = tuple(transitions)
        cls.initial = states[id(first)]

    def __init__(self, *, dt: 'unifloat | None' = None, **conditions: 'NumLike | after'):
        """Creates the state signal of the state machine.

        Arguments:
            dt: Time step of the timers (after, time_in), by default the
                period of the task
            conditions: Condition of each transition by its name: a value,
                number or after
        """
        if not self.states:
            raise TypeError(f"{type(self).__name__} declares no states")
        names = [t.name for t in self.transitions]
        missing = [n for n in names if n not in conditions]
        unknown = [n for n in conditions if n not in names]
        if missing or unknown:
            raise TypeError(f"{type(self).__name__}: " +
                            '; '.join(([f"no condition for {', '.join(missing)}"] if missing else []) +
                                      ([f"no transition {', '.join(unknown)}"] if unknown else [])))
        self.dt = dt
        self._active: list[value[int] | int] = []
        # Condition of each transition, for an after condition with the limit of its runs count
        plain: dict[Transition, value[int] | int] = {}
        timers: dict[Transition, tuple[value[int] | int, unifloat]] = {}
        for t in self.transitions:
            c = conditions[t.name]
            if isinstance(c, after):
                timers[t] = (truth(c.condition), runs_limit(c.seconds, c.dt if c.dt is not None else dt))
            else:
                plain[t] = truth(c)

        def step(previous: tuple[value[int], ...]) -> tuple[value[int], ...]:
            state = previous[0]
            counts = dict(zip(timers, previous[1:]))
            for t in self.transitions:
                if t in timers:
                    cond, limit = timers[t]
                    active = cond & self._source(state, t) & (counts[t] + 1 > limit)
                else:
                    active = plain[t] & self._source(state, t)
                self._active.append(active)
            new_state: value[int] | int = state
            for t, fires in reversed(list(zip(self.transitions, self._active))):
                new_state = iif(fires, int(t.target), new_state)
            new_state = value(new_state) if isinstance(new_state, int) else new_state
            # Runs since the condition is true in the source state, -1 otherwise:
            # counted on the new state, so the run entering the source counts
            new_counts = [iif(cond & self._source(new_state, t), counts[t] + (counts[t] < limit), -1)
                          for t, (cond, limit) in timers.items()]
            return (new_state, *new_counts)

        self.state: value[int] = feedback(step, init=(int(self.initial),) + (-1,) * len(timers))[0]
        self._previous = delay(self.state, init=-1)

    @staticmethod
    def _source(state: value[int], t: Transition) -> value[int]:
        """True if state is a source of t"""
        if t.sources is None:
            return state != int(t.target)
        return StateMachine._in(state, t.sources)

    @staticmethod
    def _in(state: value[int], states: 'tuple[State, ...]') -> value[int]:
        result = state == int(states[0])
        for s in states[1:]:
            result = result | (state == int(s))
        return result

    def _check(self, s: State) -> State:
        if not any(s is x for x in self.states):
            raise ValueError(f"{s!r} is no state of {type(self).__name__}")
        return s

    def in_(self, *states: State) -> value[int]:
        """True if the state is one of states"""
        return self._in(self.state, tuple(self._check(s) for s in states))

    def entered(self, s: State) -> value[int]:
        """True at the run at which s became the state (also at the first run
        for the initial state)"""
        self._check(s)
        return (self.state == int(s)) & (self._previous != int(s))

    def exited(self, s: State) -> value[int]:
        """True at the run at which s was left"""
        self._check(s)
        return (self._previous == int(s)) & (self.state != int(s))

    def fired(self, t: Transition) -> value[int]:
        """True at the runs at which transition t fired"""
        if not any(t is x for x in self.transitions):
            raise ValueError(f"{t!r} is no transition of {type(self).__name__}")
        index = self.transitions.index(t)
        result = self._active[index]
        for higher in self._active[:index]:
            result = result > higher  # Not blocked by a transition of higher priority
        return as_bool(result)

    def time_in(self, s: State) -> value[float]:
        """Time in seconds since s became the state, 0.0 if s is not the state"""
        self._check(s)
        stayed = (self.state == int(s)) & (self._previous == int(s))
        runs = feedback(lambda n: iif(stayed, n + (n < _MAX_RUNS), 0), init=0)
        return to_float(runs) * (self.dt if self.dt is not None else dt)

    @overload
    def select(self, outputs: Mapping[State, uniint], default: uniint | None = None) -> value[int]: ...  # type: ignore[overload-overlap]  # pyright: ignore[reportOverlappingOverload]
    @overload
    def select(self, outputs: Mapping[State, NumLike], default: NumLike | None = None) -> value[float]: ...
    @overload
    def select(self, outputs: Mapping[State, TContainer], default: TContainer | None = None) -> TContainer: ...
    def select(self, outputs: Mapping[State, Any], default: Any = None) -> Any:
        """Output by state, e.g. pump.select({Pump.RUN: u}, default=0.0). All
        outputs are computed, the one of the state is selected (see iif).
        The result is a float value if any output is a float.

        Arguments:
            outputs: Output for each state: numbers, values, vectors, tensors
                or arrays
            default: Output for the states missing in outputs

        Returns:
            The output of the state
        """
        items = [(self._check(s), v) for s, v in outputs.items()]
        if default is None:
            missing = [s for s in self.states if all(s is not x for x, _ in items)]
            if missing:
                raise ValueError(f"No output for {missing}, add them or a default")
            result = items.pop()[1]
        else:
            result = default
        for s, v in reversed(items):
            result = iif(self.state == int(s), v, result)
        if isinstance(result, int | float):
            return value_from_number(result)  # Single state machine
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(repr(t) for t in self.transitions)})"
