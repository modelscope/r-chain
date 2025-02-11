
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from tqdm.auto import tqdm


def thread_executor(max_workers: int = 8,
                    disable_tqdm: bool = False,
                    tqdm_desc: str = None):

    def decorator(func):
        @wraps(func)
        def wrapper(iterable, *args, **kwargs):
            results = []
            # Create a tqdm progress bar with the total number of items to process
            with tqdm(
                    unit_scale=True,
                    unit_divisor=1024,
                    initial=0,
                    total=len(iterable),
                    desc=tqdm_desc or f'Processing {len(iterable)} items',
                    disable=disable_tqdm,
            ) as pbar:
                # Define a wrapper function to update the progress bar
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = {
                        executor.submit(func, item, *args, **kwargs): item
                        for item in iterable
                    }

                    # Update the progress bar as tasks complete
                    for future in as_completed(futures):
                        pbar.update(1)
                        results.append(future.result())
            return results

        return wrapper

    return decorator