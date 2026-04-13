import multiprocessing

def evenly_distribute_jobs(n_items, n_jobs): 
    n_cpu = multiprocessing.cpu_count()
    if n_jobs in [None, 1]: 
        batch_size = 'auto'
    elif n_jobs == -1: 
        batch_size = n_items // n_cpu
    elif n_jobs < -1  and -n_jobs < n_cpu: 
        batch_size = n_items // n_cpu + n_jobs
    else: 
        batch_size = 1 // n_jobs

    if batch_size == 0: 
        batch_size = 1
    return batch_size

def uniquify_lol(list_of_lists):
    """Given a list of lists, return a sorted list of the unique elements in the inner lists"""
    u = set().union(*[set(_) for _ in list_of_lists])
    return sorted(list(u))


