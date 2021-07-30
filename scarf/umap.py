# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Modified file
#
# License: BSD 3 clause

import locale
from .utils import logger

locale.setlocale(locale.LC_NUMERIC, "C")

__all__ = ["fit_transform"]


def calc_dens_map_params(graph, dists):
    import numpy as np

    n_vertices = graph.shape[0]
    mu_sum = np.zeros(n_vertices, dtype=np.float32)
    ro = np.zeros(n_vertices, dtype=np.float32)
    head = graph.row
    tail = graph.col
    for i in range(len(head)):
        j = head[i]
        k = tail[i]

        D = dists[j, k] * dists[j, k]  # match sq-Euclidean used for embedding
        mu = graph.data[i]

        ro[j] += mu * D
        ro[k] += mu * D
        mu_sum[j] += mu
        mu_sum[k] += mu

    epsilon = 1e-8
    ro = np.log(epsilon + (ro / mu_sum))
    R = (ro - np.mean(ro)) / np.std(ro)
    return mu_sum, R


def simplicial_set_embedding(
    g,
    embedding,
    n_epochs,
    a,
    b,
    random_seed,
    gamma,
    initial_alpha,
    negative_sample_rate,
    densmap_kwds,
    parallel,
    nthreads,
    verbose,
):
    import numba
    from threadpoolctl import threadpool_limits
    from umap.umap_ import make_epochs_per_sample
    from umap.layouts import optimize_layout_euclidean
    from sklearn.utils import check_random_state
    import numpy as np
    import warnings
    from .utils import tqdm_params

    # g.data[g.data < (g.data.max() / float(n_epochs))] = 0.0
    # g.eliminate_zeros()
    epochs_per_sample = make_epochs_per_sample(g.data, n_epochs)
    logger.trace("calculated epochs_per_sample")
    rng_state = (
        check_random_state(random_seed)
        .randint(np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3)
        .astype(np.int64)
    )
    # Since threadpool_limits doesnt work well with numba. We will use numba's set_num_threads to limit threads
    if numba.config.NUMBA_NUM_THREADS > nthreads:
        numba.set_num_threads(nthreads)

    if densmap_kwds != {}:
        with threadpool_limits(limits=nthreads):
            mu_sum, R = calc_dens_map_params(g, densmap_kwds["knn_dists"])
        densmap_kwds["mu_sum"] = mu_sum
        densmap_kwds["R"] = R
        densmap_kwds["mu"] = g.data
        densmap = True
        logger.trace("calculated densmap params")
    else:
        densmap = False

    # tqdm will be activated if https://github.com/lmcinnes/umap/pull/739
    # is merged and when it is released
    tqdm_params = dict(tqdm_params)
    tqdm_params["desc"] = "Training UMAP"

    with threadpool_limits(limits=nthreads):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            embedding = optimize_layout_euclidean(
                embedding,
                embedding,
                g.row,
                g.col,
                n_epochs,
                g.shape[1],
                epochs_per_sample,
                a,
                b,
                rng_state,
                gamma,
                initial_alpha,
                negative_sample_rate,
                parallel=parallel,
                verbose=verbose,
                densmap=densmap,
                densmap_kwds=densmap_kwds,
                # tqdm_kwds=tqdm_params,
            )
    return embedding


# def fuzzy_simplicial_set(g, set_op_mix_ratio):
#     tg = g.transpose()
#     prod = g.multiply(tg)
#     res = set_op_mix_ratio * (g + tg - prod) + (1.0 - set_op_mix_ratio) * prod
#     res.eliminate_zeros()
#     return res.tocoo()


def fit_transform(
    graph,
    ini_embed,
    spread,
    min_dist,
    # set_op_mix_ratio,
    n_epochs,
    random_seed,
    repulsion_strength,
    initial_alpha,
    negative_sample_rate,
    densmap_kwds,
    parallel,
    nthreads,
    verbose,
):
    from umap.umap_ import find_ab_params

    a, b = find_ab_params(spread, min_dist)
    logger.trace("Found ab params")
    # sym_graph = fuzzy_simplicial_set(graph, set_op_mix_ratio)

    embedding = simplicial_set_embedding(
        graph,
        ini_embed,
        n_epochs,
        a,
        b,
        random_seed,
        repulsion_strength,
        initial_alpha,
        negative_sample_rate,
        densmap_kwds,
        parallel,
        nthreads,
        verbose,
    )
    return embedding, a, b
