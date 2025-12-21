import copapy as cp

@cp.jit
def calculation(x: float, y: float) -> float:
    return sum(x ** 2 + y ** 2 + i for i in range(10))


MASK = (1 << 31) - 1  # 0x7FFFFFFF


def rotl31(x: int, r: int) -> int:
    r %= 31
    return ((x << r) | (x >> (31 - r))) & MASK


def slow_31bit_int_list_hash(data: list[int], rounds: int = 5)-> int:
    """
    Intentionally slow hash using only 31-bit integer operations.
    Input:  list[int]
    Output: 31-bit integer
    """

    # 31-bit initial state (non-zero)
    state = 0x1234567 & MASK

    # Normalize input into 31-bit space
    data = [abs(x) & MASK for x in data]

    for r in range(rounds):
        for i, x in enumerate(data):
            # Mix index, round, and data
            state ^= (x + i + r) & MASK

            # Nonlinear mixing (carefully kept 31-bit)
            state = (state * 1103515245) & MASK
            state ^= (state >> 13)
            state = (state * 12345) & MASK

            # Data-dependent rotation (forces serial dependency)
            rot = (x ^ state) % 31
            state = rotl31(state, rot)

        # Cross-round diffusion
        state ^= (state >> 11)
        state = (state * 1664525) & MASK
        state ^= (state >> 17)

    return state


def test_decorator():
    sumv = 0
    y = 5.7
    for i in range(2000):
        x = i * 2.5
        sumv = calculation(x, y) + sumv

    assert abs(sumv - 166542418649.28778) < 1e14, sumv

def test_hash():
    nums = [12, 99, 2024]
    h_ref = slow_31bit_int_list_hash(nums)
    print(h_ref)

    h = cp.jit(slow_31bit_int_list_hash)(nums)
    print(h)
    assert h == h_ref
