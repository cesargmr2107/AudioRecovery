import math
import random

import numpy


def gen_identity_matrix(n):
    identity = numpy.zeros(shape=(n, n))
    for i in range(n):
        identity[n - i - 1] = int_to_bin(2 ** i, n)
    return identity


def int_to_bin(number, n_bits):
    bin_list = [int(x) for x in bin(number)[2:]]
    size = len(bin_list)
    while n_bits > size:
        bin_list.insert(0, 0)
        size += 1
    return numpy.array(bin_list).astype('int')


def bin_to_int(bin_ndarray: numpy.ndarray):
    value = 0
    bin_list = bin_ndarray.astype(dtype='int').tolist()
    for i in range(len(bin_list)):
        digit = bin_list.pop()
        if digit == 1:
            value = value + pow(2, i)
    return value


def systemize_matrix(matrix: numpy.ndarray):
    # Build systemized matrix and save transposed Q
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    systemized = gen_identity_matrix(n_rows)
    transposed_q = []
    for col_index in range(n_cols):
        transposed_col = matrix[:, col_index]
        col_value = bin_to_int(transposed_col)
        if not math.log(col_value, 2).is_integer():
            systemized = numpy.c_[systemized, transposed_col]
            transposed_q.append(transposed_col)

    # Get changes
    changes = []
    for col_index in range(n_cols):
        original_col_value = bin_to_int(matrix[:, col_index])
        if original_col_value != bin_to_int(systemized[:, col_index]):
            look_up_index = 0
            while look_up_index < n_cols and original_col_value != bin_to_int(systemized[:, look_up_index]):
                look_up_index += 1
            changes.append((col_index,look_up_index))

    return systemized, numpy.array(transposed_q), changes

def gen_random_errors(vector, n_errors):
    n_elements = len(vector)
    error_vector = [0] * (n_elements - n_errors)
    error_vector.extend([1] * n_errors)
    random.shuffle(error_vector)
    vector_with_errors = fix_binary_numpy_array(numpy.array(error_vector) + numpy.array(vector))
    return vector_with_errors, error_vector


def fix_binary_numpy_array(array: numpy.ndarray):
    for bit_index in range(len(array)):
        array[bit_index] = 0 if array[bit_index] % 2 == 0 else 1
    return array


def show_numpy_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])
    print()
