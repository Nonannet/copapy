import copapy as cp

# Arm lengths
l1, l2 = 1.8, 2.0

# Target position
target = cp.vector([0.7, 0.7])

# Learning rate for iterative adjustment
alpha = 0.1


def forward_kinematics(theta1: cp.value[float] | float, theta2: cp.value[float] | float) -> tuple[cp.vector[float], cp.vector[float]]:
    """Return positions of joint and end-effector."""
    joint = cp.vector([l1 * cp.cos(theta1), l1 * cp.sin(theta1)])
    end_effector = joint + cp.vector([l2 * cp.cos(theta1 + theta2),
                                    l2 * cp.sin(theta1 + theta2)])
    return joint, end_effector


def test_two_arms():    
    target_vec = cp.vector(target)
    theta = cp.vector([cp.value(0.0), cp.value(0.0)])

    joint = cp.vector([0.0, 0.0])
    effector = cp.vector([0.0, 0.0])
    error = 0.0

    # Iterative IK
    for _ in range(48):
        joint, effector = forward_kinematics(theta[0], theta[1])
        error = ((target_vec - effector) ** 2).sum()

        grad_vec = cp.grad(error, theta)
        theta -= alpha * grad_vec

    tg = cp.Target()
    tg.compile(error, theta, joint)
    tg.run()

    print(f"Joint angles: {tg.read_value(theta)}")
    print(f"Joint position: {tg.read_value(joint)}")
    print(f"End-effector position: {tg.read_value(effector)}")
    print(f"quadratic error = {tg.read_value(error)}")

    assert tg.read_value(error) < 1e-6


if __name__ == '__main__':
    test_two_arms()