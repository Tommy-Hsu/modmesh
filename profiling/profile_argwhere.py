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

        # Calculate speedup as base_time / func_time
        # For np itself, this will be 1.0
        # For slower functions, this will be < 1.0 (e.g., 0.47 means it's 47% as fast as np)
        # For faster functions (if any), this would be > 1.0
        speedup = (base_time / func['total_time'] 
                   if func['total_time'] > 0 else float('inf'))

        # Format and print the row
        print(f"{name:<30} {count:<15} {total_time_ms:>15.3f} ms {per_call_ms:>15.3f} ms {speedup:>10.2f}x")


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
def profile_argwhere_np(narr):
    return np.argwhere(narr)


@profile_function
def profile_argwhere_sa(sarr):
    return sarr.argwhere()


def main():
    # 定義要測試的陣列大小
    sizes = [
        128,            # 128
        1024,           # 1K
        8 * 1024,       # 8K
        64 * 1024,      # 64K
        512 * 1024,     # 512K
        4 * 1024 * 1024,     # 4M
        32 * 1024 * 1024     # 32M
    ]
    
    it = 50
    
    for N in sizes:
        print(f"\n{'=' * 95}")
        print(f"{f'測試陣列大小: {N:,} 元素':^95}")
        print(f"{'=' * 95}")
        
        modmesh.call_profiler.reset()
        for _ in range(it):
            test_data = np.arange(0, N, dtype='uint32')
            np.random.shuffle(test_data)
            profile_argwhere_np(test_data)
            profile_argwhere_sa(np2sa(test_data))
        
        # root_stat = modmesh.call_profiler.stat()
        root_res = modmesh.call_profiler.result()
        # print(root_stat)
        # print(root_res)
        format_profiling_data(root_res)


if __name__ == '__main__':
    main()
