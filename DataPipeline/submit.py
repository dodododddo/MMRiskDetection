import concurrent.futures 

def f(x):
    return x

with concurrent.futures.ProcessPoolExecutor(max_workers=1000) as executor:
    futures = [executor.submit(f, i) for i in range(1000)]
    answers = [future.result() for future in concurrent.futures.as_completed(futures)]
    
print(answers)