import math
import multiprocessing


def effective_workers(n_jobs):
    """Normalize a joblib-style n_jobs into a positive worker count.

    None -> 1, positive -> itself, -1 -> all cores, -2 -> all but one, ...
    (negative values follow the joblib convention n_cpu + 1 + n_jobs).
    """
    if n_jobs is None:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() + 1 + n_jobs
    return max(1, n_jobs)


def evenly_distribute_jobs(n_items, n_jobs, oversubscription=4):
    """Number of items per parallel task so that n_items is spread over the workers.

    Returns ceil(n_items / (oversubscription * n_workers)), at least 1. With
    oversubscription=1 every worker gets exactly one task; a larger value gives
    the scheduler slack to balance uneven items (a few large graphs in a mostly
    small dataset) while still keeping the number of tasks independent of n_items.
    """
    n_workers = effective_workers(n_jobs)
    n_tasks = max(1, int(oversubscription)) * n_workers
    return max(1, math.ceil(n_items / n_tasks))


def uniquify_lol(list_of_lists):
    """Given a list of lists, return a sorted list of the unique elements in the inner lists"""
    u = set().union(*[set(_) for _ in list_of_lists])
    return sorted(list(u))


