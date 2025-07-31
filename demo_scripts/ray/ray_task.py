
import time
import ray
import logging

@ray.remote
def ray_sum_of_squares(n):
    start = time.time()
    total = sum(i*i for i in range(n))
    total_time = time.time() - start
    return total, total_time

def ray_execution(fun,inputs):
    start_time = time.time()
    futures = [ray_sum_of_squares.remote(input) for input in inputs]
    ray_result = ray.get(futures)
    print(f"total time take : {time.time() - start_time}s")
    return ray_result

if __name__ == "__main__":
    inputs = [ 4 ,10_000_000,20_000_000,30_000_000,40_000_000,50_000_000,60_000_000, 70_000_000]
    ray_result = ray_execution(ray_sum_of_squares,inputs)



