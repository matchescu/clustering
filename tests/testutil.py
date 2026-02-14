import itertools


def is_partition_over(all_refs, clusters):
    all_elements = frozenset(all_refs)
    covered = frozenset.union(*clusters) if clusters else frozenset()

    if all_elements != covered:
        missing = all_elements - covered
        extra = covered - all_elements
        msg = f"Partition incomplete. Missing: {missing}"
        if extra:
            msg += f", Extra: {extra}"
        raise AssertionError(msg)

    # Check for overlaps
    for a, b in itertools.combinations(clusters, 2):
        overlap = a & b
        if overlap:
            raise AssertionError(f"Clusters overlap: {a} ∩ {b} = {overlap}")

    return True
