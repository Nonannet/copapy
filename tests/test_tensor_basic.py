#!/usr/bin/env python3
"""Basic tests for the tensor class."""

import copapy as cp

def test_tensor_basic():
    # Test 1: Create a scalar tensor
    print("Test 1: Scalar tensor")
    t0 = cp.tensor(42)
    print(f"Scalar tensor: {t0}")
    print(f"Shape: {t0.shape}, ndim: {t0.ndim}")
    assert t0.shape == ()
    assert t0 == 42
    print()

    # Test 2: Create a 1D tensor from list
    print("Test 2: 1D tensor")
    t1 = cp.tensor([1, 2, 3, 4, 5])
    print(f"1D tensor: shape={t1.shape}, ndim={t1.ndim}")
    print(f"Elements: {[t1[i] for i in range(len(t1))]}")
    assert t1.shape == (5,)
    assert t1.ndim == 1
    assert t1[0] == 1
    print()

    # Test 3: Create a 2D tensor (matrix)
    print("Test 3: 2D tensor")
    t2 = cp.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"2D tensor: shape={t2.shape}, ndim={t2.ndim}")
    print(f"Element [0,1]: {t2[0, 1]}")
    print(f"Row 1: {t2[1]}")
    assert t2.shape == (2, 3)
    assert t2.ndim == 2
    assert t2[0, 1] == 2

    assert t2[1][2] == 6
    print()

    # Test 4: Create a 3D tensor
    print("Test 4: 3D tensor")
    t3 = cp.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"3D tensor: shape={t3.shape}, ndim={t3.ndim}")
    print(f"Element [0,1,0]: {t3[0, 1, 0]}")
    assert t3.shape == (2, 2, 2)
    assert t3.ndim == 3
    assert t3[0, 1, 0] == 3
    print()

    # Test 6: Broadcasting with scalar
    print("Test 6: Broadcasting with scalar")
    t = cp.tensor([1.0, 2.0, 3.0])
    result = t * 2.0
    print(f"tensor * 2.0: shape={result.shape}")
    print(f"Elements: {[result[i] for i in range(len(result))]}")
    assert result.shape == (3,)
    assert result[0] == 2.0
    assert result[1] == 4.0
    print()

    # Test 6b: Broadcasting with different dimensions
    print("Test 6b: Broadcasting with different dimensions")
    # 2D tensor + 1D tensor
    t2d = cp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t1d = cp.tensor([10.0, 20.0, 30.0])
    result_2d_1d = t2d + t1d
    print(f"2D tensor {t2d.shape} + 1D tensor {t1d.shape} = shape {result_2d_1d.shape}")
    print(f"Elements: {[[result_2d_1d[i, j] for j in range(3)] for i in range(2)]}")
    assert result_2d_1d.shape == (2, 3)
    assert result_2d_1d[0, 0] == 11.0
    assert result_2d_1d[1, 2] == 36.0
    
    # 3D tensor + 2D tensor
    t3d = cp.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    t2d_broadcast = cp.tensor([[100.0, 200.0], [300.0, 400.0]])
    result_3d_2d = t3d + t2d_broadcast
    print(f"3D tensor {t3d.shape} + 2D tensor {t2d_broadcast.shape} = shape {result_3d_2d.shape}")
    assert result_3d_2d.shape == (2, 2, 2)
    assert result_3d_2d[0, 0, 0] == 101.0
    assert result_3d_2d[1, 1, 1] == 408.0
    
    # 3D tensor + 1D tensor
    t1d_broadcast = cp.tensor([1.0, 2.0])
    result_3d_1d = t3d + t1d_broadcast
    print(f"3D tensor {t3d.shape} + 1D tensor {t1d_broadcast.shape} = shape {result_3d_1d.shape}")
    assert result_3d_1d.shape == (2, 2, 2)
    assert result_3d_1d[0, 0, 0] == 2.0
    assert result_3d_1d[1, 1, 1] == 10.0
    print()

    # 3D tensor + vector
    t1d_broadcast = cp.vector([1.0, 2.0])
    result_3d_1d = t3d + t1d_broadcast
    print(f"3D tensor {t3d.shape} + 1D tensor {t1d_broadcast.shape} = shape {result_3d_1d.shape}")
    assert result_3d_1d.shape == (2, 2, 2)
    assert result_3d_1d[0, 0, 0] == 2.0
    assert result_3d_1d[1, 1, 1] == 10.0
    print()

    # Test 6c: Element-wise operations with different dimensions
    print("Test 6c: Element-wise operations with different dimensions")
    a2d = cp.tensor([[1.0, 2.0], [3.0, 4.0]])
    b2d = cp.tensor([[2.0, 3.0], [4.0, 5.0]])
    c2d = a2d * b2d
    print(f"2D * 2D: shape={c2d.shape}")
    print(f"Elements: {[[c2d[i, j] for j in range(2)] for i in range(2)]}")
    assert c2d.shape == (2, 2)
    assert c2d[0, 0] == 2.0
    assert c2d[1, 1] == 20.0
    
    # 3D - 2D
    t3d_sub = cp.tensor([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]])
    t2d_sub = cp.tensor([[1.0, 2.0], [3.0, 4.0]])
    result_sub = t3d_sub - t2d_sub
    print(f"3D - 2D: shape={result_sub.shape}")
    assert result_sub.shape == (2, 2, 2)
    assert result_sub[0, 0, 0] == 9.0
    assert result_sub[1, 1, 1] == 76.0
    print()

    # Test 7: Reshape
    print("Test 7: Reshape")
    t = cp.tensor([1, 2, 3, 4, 5, 6])
    print(f"Original: shape={t.shape}")
    t_reshaped = t.reshape(2, 3)
    print(f"Reshaped to (2, 3): shape={t_reshaped.shape}")
    print(f"Element [1,2]: {t_reshaped[1, 2]}")
    assert t_reshaped.shape == (2, 3)
    assert t_reshaped[1, 2] == 6
    assert t_reshaped[0, 0] == 1
    print()

    # Test 8: Flatten
    print("Test 8: Flatten")
    t = cp.tensor([[1, 2, 3], [4, 5, 6]])
    flat = t.flatten()
    print(f"Original: shape={t.shape}")
    print(f"Flattened: shape={flat.shape}")
    print(f"Elements: {[flat[i] for i in range(len(flat))]}")
    assert flat.shape == (6,)
    assert flat[0] == 1
    assert flat[5] == 6
    print()

    # Test 9: Transpose
    print("Test 9: Transpose")
    t = cp.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Original: shape={t.shape}")
    t_t = t.transpose()
    print(f"Transposed: shape={t_t.shape}")
    print(f"Element [2,1]: {t_t[2, 1]}")
    assert t_t.shape == (3, 2)
    assert t_t[2, 1] == 6
    print()

    # Test 10: Sum operations
    print("Test 10: Sum operations")
    t = cp.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Original: shape={t.shape}")
    total = t.sum()
    print(f"Sum all: {total}")
    sum_axis0 = t.sum(axis=0)
    print(f"Sum along axis 0: shape={sum_axis0.shape}")
    sum_axis1 = t.sum(axis=1)
    print(f"Sum along axis 1: shape={sum_axis1.shape}")
    assert total == 21
    assert sum_axis0.shape == (3,)
    assert sum_axis0[0] == 5
    assert sum_axis1.shape == (2,)
    assert sum_axis1[1] == 15
    print()

    # Test 10b: Sum with multiple axes and keepdims
    print("Test 10b: Sum with multiple axes and keepdims")
    t3d = cp.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"Original 3D tensor: shape={t3d.shape}")
    
    # Sum along multiple axes
    sum_axes_0_2 = t3d.sum(axis=(0, 2))
    print(f"Sum along axes (0, 2): shape={sum_axes_0_2.shape}")
    assert sum_axes_0_2.shape == (2,), f"Expected (2,), got {sum_axes_0_2.shape}"
    assert sum_axes_0_2[0] == 1 + 2 + 5 + 6  # Elements from [0,:,*] and [1,:,*]
    assert sum_axes_0_2[1] == 3 + 4 + 7 + 8
    print(f"Values: {[sum_axes_0_2[i] for i in range(len(sum_axes_0_2))]}")
    
    # Sum with keepdims
    sum_keepdims = t3d.sum(axis=1, keepdims=True)
    print(f"Sum along axis 1 with keepdims: shape={sum_keepdims.shape}")
    assert sum_keepdims.shape == (2, 1, 2), f"Expected (2, 1, 2), got {sum_keepdims.shape}"
    
    # Sum multiple axes with keepdims
    sum_multi_keepdims = t3d.sum(axis=(0, 2), keepdims=True)
    print(f"Sum along axes (0, 2) with keepdims: shape={sum_multi_keepdims.shape}")
    assert sum_multi_keepdims.shape == (1, 2, 1), f"Expected (1, 2, 1), got {sum_multi_keepdims.shape}"
    
    # Sum all axes with keepdims
    sum_all_keepdims = t3d.sum(keepdims=True)
    print(f"Sum all with keepdims: shape={sum_all_keepdims.shape}")
    assert sum_all_keepdims.shape == (1, 1, 1), f"Expected (1, 1, 1), got {sum_all_keepdims.shape}"
    assert sum_all_keepdims[0, 0, 0] == 36  # Sum of all elements
    print()

    # Test 11: Factory functions
    print("Test 11: Factory functions")
    z = cp.zeros((2, 3))
    print(f"zeros((2, 3)): shape={z.shape}")
    o = cp.ones((3, 2))
    print(f"ones((3, 2)): shape={o.shape}")
    e = cp.eye(3)
    print(f"eye(3): shape={e.shape}")
    ar = cp.arange(0, 10, 2)
    print(f"arange(0, 10, 2): shape={ar.shape}")
    assert z.shape == (2, 3)
    assert z[0, 0] == 0
    assert o.shape == (3, 2)
    assert o[1, 1] == 1
    print()

    # Test 12: Size and properties
    print("Test 12: Size and properties")
    t = cp.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"Shape: {t.shape}")
    print(f"ndim: {t.ndim}")
    print(f"size: {t.size()}")
    assert t.shape == (2, 2, 2)
    assert t.ndim == 3
    print()

def test_tensor_slicing():
    print("Test Numpy-style slicing")
    t = cp.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    print(f"Original tensor: shape={t.shape}")

    slice1 = t[1]
    print(f"t[1]: {slice1}, shape={slice1.shape}")
    assert slice1.shape == (3,)
    assert slice1[0] == 40

    slice2 = t[:, 2]
    print(f"t[:, 2]: {slice2}, shape={slice2.shape}")
    assert slice2.shape == (3,)
    assert slice2[1] == 60

    slice3 = t[0:2, 1:3]
    print(f"t[0:2, 1:3]: {slice3}, shape={slice3.shape}")
    assert slice3.shape == (2, 2)
    assert slice3[0, 0] == 20

    slice4 = t[-1, :]
    print(f"t[-1, :]: {slice4}, shape={slice4.shape}")
    assert slice4.shape == (3,)
    assert slice4[2] == 90
    print()

if __name__ == "__main__":
    test_tensor_basic()
    print("All tests completed!")

