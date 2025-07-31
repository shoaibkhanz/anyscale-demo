
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def process_pool(func, inputs):
    import time

    start_time = time.time()
    with ProcessPoolExecutor() as process_executor:
        futures = {process_executor.submit(func, input) for input in inputs}

        for future in as_completed(futures):
            result, execution_time = future.result()
            print(f"{result} -> time taken {execution_time:.4f}s")
    print(f"total time taken {(time.time() - start_time):.4f}s")
# now we create a simple sum_of_squares function
def sum_of_squares(n):
    start = time.time()
    total = sum(i*i for i in range(n))
    total_time = time.time() - start
    return total, total_time

if __name__ == "__main__":
    inputs = [ 4 ,10_000_000,20_000_000,30_000_000,40_000_000,50_000_000,60_000_000, 70_000_000]
    process_pool(sum_of_squares, inputs)



