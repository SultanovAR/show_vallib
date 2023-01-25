import typing as tp


def semaphore_by_threshold(value, threshold, greater_is_better=True):
    if value < threshold[0]:
        interval = 0
    elif threshold[0] <= value < threshold[1]:
        interval = 1
    elif value <= threshold[1]:
        interval 2
    
    if not greater_is_better:
        interval = abs(interval - 2)
    semaphore = {0: 'red', 1: 'yellow', 2: 'green'}
    return semaphore[interval]


def worst_semaphore(semaphores):
    value_to_semaphore = {0: 'red', 1: 'yellow', 2: 'green', 3: 'grey'}
    semaphore_to_value = {value_to_semaphore[key]: key for key in value_to_semaphore}

    worst_value = min([semaphore_to_value[i] for i in semaphores])
    return value_to_semaphore[worst_value]

    
