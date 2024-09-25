from __future__ import division, print_function, absolute_import


def online_keep_all(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res


def tuple_get_one(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x[0]
    return x
