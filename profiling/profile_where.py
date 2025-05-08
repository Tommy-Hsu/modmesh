import functools

import numpy as np

import modmesh


def format_profiling_data(profile_data):
    functions = profile_data['children']

    np_func = next((func for func in functions if "np" in func['name']), None)

    if not np_func:
        print("Warning: 'profile_argwhere_np' not found in profiling data, "
              "using first function as base")
        np_func = functions[0]

    base_time = np_func['total_time']

    print(f"{'Function Name':<30} {'Call Count':<15} {'Total Time (ms)':<20} "
          f"{'Per Call (ms)':<20} {'Speedup':<10}")
    print('-' * 95)

    # Sort by total time
    for func in sorted(functions, key=lambda x: x['total_time']):
        name = func['name']
        count = func['count']
        total_time_ms = func['total_time'] * 1000  # Convert to milliseconds
        per_call_ms = total_time_ms / count if count > 0 else 0

        speedup = (base_time / func['total_time']
                   if func['total_time'] > 0 else float('inf'))

        # Format and print the row
        print(
            f"{name:<30} {count:<15} {total_time_ms:>15.3f} ms "
            f"{per_call_ms:>15.3f} ms {speedup:>10.2f}x"
        )


def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = modmesh.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result
    return wrapper


def np2sa(narr):
    if narr.dtype == np.bool_:
        return modmesh.SimpleArrayBool(array=narr)
    elif np.isdtype(narr.dtype, np.int8):
        return modmesh.SimpleArrayInt8(array=narr)
    elif np.isdtype(narr.dtype, np.int16):
        return modmesh.SimpleArrayInt16(array=narr)
    elif np.isdtype(narr.dtype, np.int32):
        return modmesh.SimpleArrayInt32(array=narr)
    elif np.isdtype(narr.dtype, np.int64):
        return modmesh.SimpleArrayInt64(array=narr)
    elif np.isdtype(narr.dtype, np.uint8):
        return modmesh.SimpleArrayUint8(array=narr)
    elif np.isdtype(narr.dtype, np.uint16):
        return modmesh.SimpleArrayUint16(array=narr)
    elif np.isdtype(narr.dtype, np.uint32):
        return modmesh.SimpleArrayUint32(array=narr)
    elif np.isdtype(narr.dtype, np.uint64):
        return modmesh.SimpleArrayUint64(array=narr)
    elif np.isdtype(narr.dtype, np.float32):
        return modmesh.SimpleArrayFloat32(array=narr)
    elif np.isdtype(narr.dtype, np.float64):
        return modmesh.SimpleArrayFloat64(array=narr)
    else:
        raise ValueError("Unsupported data type for SimpleArray conversion.")


@profile_function
def profile_argwhere_np_gt(narr, value):
    return np.argwhere(narr > value)


@profile_function
def profile_argwhere_sa_gt(sarr, value):
    return sarr.argwhere(lambda x: x > value)


@profile_function
def profile_argwhere_np(narr):
    return np.argwhere(narr)


@profile_function
def profile_argwhere_sa(sarr):
    return sarr.argwhere()


@profile_function
def profile_argwhere_np_range(narr, small_value, large_value):
    narr = (narr > small_value) & (narr < large_value)
    return np.argwhere(narr)


@profile_function
def profile_argwhere_sa_range(sarr, small_value, large_value):
    sarr_filtered = sarr.argwhere(lambda x: small_value < x < large_value)
    return sarr_filtered


@profile_function
def profile_argwhere_np_complex(narr):
    return np.argwhere(
            ((narr > 10) & (narr <= 50)) |
            (narr == 7) |
            ((narr != 0) & (narr < -5)) |
            ((narr >= 100) & (narr <= 200) & (narr != 150)) |
            ((narr % 3 == 0) & (narr % 2 != 0) & (narr > 0) & (narr < 30))
    )


@profile_function
def profile_argwhere_sa_complex(sarr):
    return sarr.argwhere(
        lambda x: (
            ((x > 10) & (x <= 50)) |
            (x == 7) |
            ((x != 0) & (x < -5)) |
            ((x >= 100) & (x <= 200) & (x != 150)) |
            ((x % 3 == 0) & (x % 2 != 0) & (x > 0) & (x < 30))
        )
    )


def main():

    sizes = [
        128,            # 128
        1024,           # 1K
        8 * 1024,       # 8K
        64 * 1024,      # 64K
        512 * 1024,     # 512K
        4 * 1024 * 1024,     # 4M
        32 * 1024 * 1024     # 32M
    ]

    dtype_range_max = {
        'int8': 127,
        'int16': 32767,
        'int32': 2147483647,
        'int64': 9223372036854775807,
        'uint8': 255,
        'uint16': 65535,
        'uint32': 4294967295,
        'uint64': 18446744073709551615,
        'float32': 3.4028235e+38,
        'float64': 1.0e200 
    }

    dtype_range_min = {
        'int8': -128,
        'int16': -32768,
        'int32': -2147483648,
        'int64': -9223372036854775808,
        'uint8': 0,
        'uint16': 0,
        'uint32': 0,
        'uint64': 0,
        'float32': -3.4028235e+38,
        'float64': -1.0e200 
    }

    dtype = [
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
        'uint32',
        'uint64',
        'float32',
        'float64'
    ]
    
    it = 50

    custom_dtype = dtype[9]
    
    for N in sizes:
        print(f"\n{'=' * 95}")
        print(f"{f'測試陣列大小: {N:,} 元素':^95}")
        print(f"{'=' * 95}")
        
        modmesh.call_profiler.reset()
        for _ in range(it):
            test_data = np.random.uniform(
                low=dtype_range_min[custom_dtype],
                high=dtype_range_max[custom_dtype],
                size=N
            ).astype(custom_dtype)
            profile_argwhere_np(test_data)
            profile_argwhere_sa(np2sa(test_data))
        
        root_res = modmesh.call_profiler.result()
        format_profiling_data(root_res)

    for _ in range(5):
        print()

    for N in sizes:
        print(f"\n{'=' * 95}")
        print(f"{f'測試陣列大小: {N:,} 元素':^95}")
        print(f"{'=' * 95}")
        
        modmesh.call_profiler.reset()
        for _ in range(it):
            test_data = np.random.uniform(
                low=dtype_range_min[custom_dtype],
                high=dtype_range_max[custom_dtype],
                size=N
            ).astype(custom_dtype)
            profile_argwhere_np_gt(test_data, dtype_range_min[custom_dtype]/2)
            profile_argwhere_sa_gt(
                np2sa(test_data), dtype_range_min[custom_dtype] / 2
            )
        
        root_res = modmesh.call_profiler.result()
        format_profiling_data(root_res)

    for _ in range(5):
        print()

    for N in sizes:
        print(f"\n{'=' * 95}")
        print(f"{f'測試陣列大小: {N:,} 元素':^95}")
        print(f"{'=' * 95}")
        
        modmesh.call_profiler.reset()
        for _ in range(it):
            test_data = np.random.uniform(
                low=dtype_range_min[custom_dtype],
                high=dtype_range_max[custom_dtype],
                size=N
            ).astype(custom_dtype)
            profile_argwhere_np_range(
                test_data,
                dtype_range_min[custom_dtype] / 2,
                dtype_range_max[custom_dtype] / 2
            )
            profile_argwhere_sa_range(
                np2sa(test_data),
                dtype_range_min[custom_dtype] / 4,
                dtype_range_max[custom_dtype] / 2
            )
        
        root_res = modmesh.call_profiler.result()
        format_profiling_data(root_res)

    for _ in range(5):
        print()

    for N in sizes:
        print(f"\n{'=' * 95}")
        print(f"{f'測試陣列大小: {N:,} 元素':^95}")
        print(f"{'=' * 95}")
        
        modmesh.call_profiler.reset()
        for _ in range(it):
            test_data = np.random.uniform(
                low=dtype_range_min[custom_dtype],
                high=dtype_range_max[custom_dtype],
                size=N
            ).astype(custom_dtype)
            profile_argwhere_np_complex(
                test_data
            )
            profile_argwhere_sa_complex(
                np2sa(test_data)
            )
        
        root_res = modmesh.call_profiler.result()
        format_profiling_data(root_res)


if __name__ == '__main__':
    main()
