import numpy as np

def my_function(x):
    return x*2

def main():

    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(my_function, np.arange(0, 1000000)))

    print(results)

if __name__ == "__main__":
    main()

