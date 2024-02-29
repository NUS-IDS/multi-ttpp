import pytest

import torch
import torch.utils.data as data_utils
import numpy as np

from multittpp.data import SequenceDataset, collate

#
# SETUP
#


@pytest.fixture
def seed():
    torch.manual_seed(572)


@pytest.fixture(params=[(10, 20, 1)])
def shape(request):
    return request.param


@pytest.fixture(
    params=[
        (2,),
        (2, 5),
    ]
)
def lambdas(request):
    return request.param


@pytest.fixture
def n_marks(lambdas):
    return len(lambdas)


@pytest.fixture
def dtype():
    return torch.float


@pytest.fixture
def device():
    return "cuda:3" if torch.cuda.is_available else "cpu"


@pytest.fixture
def homogenous_poisson(shape, lambdas, dtype, device, seed):
    B, N, _ = shape
    K = len(lambdas)
    # initialize the array
    # we will produce N events for each process so K*N
    x = np.zeros((B, K * N))
    k = np.zeros((B, K * N), dtype=int)
    # simulate each process one by one
    for i in range(K):
        # draw N samples from the uniform distribution
        u = np.random.random((B, N))
        # convert u to exponentially distributed intervals, cumsum to obtain time of event
        x[:, (i * N) : ((i + 1) * N)] = np.cumsum(-np.log((1 - u)) / lambdas[i], axis=1)
        k[:, (i * N) : ((i + 1) * N)] = i
    # sort the time of all events
    ix = x.argsort(axis=-1)
    x = x[np.expand_dims(np.arange(B), axis=-1).repeat(N * K, axis=-1), ix]
    k = k[np.expand_dims(np.arange(B), axis=-1).repeat(N * K, axis=-1), ix]
    # pick the first N events from the K*N simulated events
    data = {
        "timestamps": [],
        "types": [],
        "lengths": [],
        "intervals": [],
        "t_max": 0.0,
        "event_type_num": K,
    }
    for b, n in enumerate(np.random.randint(2, N, (B,))):
        # ensure that at least one simulation has N events
        if b == 8:
            n = N
        # stop each simulation batch after n events (randomly sampled)
        data["timestamps"].append(x[b, :n])
        data["types"].append(k[b, :n])
        data["lengths"].append(n)
        data["intervals"].append(np.diff(x[b, :n], prepend=0))

    data["timestamps"] = np.array(data["timestamps"], dtype=object)
    data["types"] = np.array(data["types"], dtype=object)
    data["intervals"] = np.array(data["intervals"], dtype=object)
    data["t_max"] = np.concatenate(data["timestamps"]).max()

    dset = SequenceDataset(data, device=device)

    return dset


@pytest.fixture
def dset(homogenous_poisson):
    return homogenous_poisson


@pytest.fixture
def batch(dset):
    loader = data_utils.DataLoader(dset, batch_size=4, shuffle=True, collate_fn=collate)
    return next(iter(loader))
