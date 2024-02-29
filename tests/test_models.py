import pytest

import torch
import torch.nn.functional as F
from torch.testing import make_tensor

import multittpp.models


@pytest.mark.parametrize(
    "model,attr",
    [
        (multittpp.models.InhomogeneousPoisson, None),
        (multittpp.models.Renewal, "both"),
        (multittpp.models.ModulatedRenewal, "both"),
        (multittpp.models.TriTPP, "both"),
        (multittpp.models.SplineTransformer, "both"),
    ],
)
def test_model_clamp_direction_attr(model, dset, attr):
    model = model(
        n_marks=dset.n_marks, n_events=dset.N_max, t_max_normalization=dset.t_max
    )
    if attr is None:
        assert not hasattr(model, "clamp_direction")
    else:
        assert model.clamp_direction == attr


@pytest.fixture(
    params=[
        multittpp.models.InhomogeneousPoisson,
        multittpp.models.Renewal,
        multittpp.models.ModulatedRenewal,
        multittpp.models.TriTPP,
        multittpp.models.SplineTransformer,
    ]
)
def model(request, dset):
    return request.param(
        n_marks=dset.n_marks,
        n_events=dset.N_max,
        t_max_normalization=dset.t_max,
        dropout=0,
    )


def test_model_forward(model, batch):
    x = batch.in_times
    xT = batch.last_times
    k = batch.in_types
    model.to(x)
    y1 = model.forward(x, k)
    y2, yT = model.forward(x, k, xT)
    mask = k < model.n_marks
    assert torch.allclose(y1[mask], y2[mask], atol=1e-6)
    assert xT.shape == yT.shape


def test_model_compensator(model, batch):
    y = batch.in_times
    yT = batch.last_times
    k = batch.in_types
    model.to(y)
    x1, log_intensity1 = model.compensator(y, k)
    x2, log_intensity2, xT, log_intensityT = model.compensator(y, k, yT)
    mask = k < model.n_marks
    assert torch.allclose(x1[mask], x2[mask], atol=1e-6)
    assert torch.allclose(log_intensity1[mask], log_intensity2[mask], atol=1e-6)
    assert xT.shape == yT.shape
    assert xT.shape == log_intensityT.shape


def test_model_inverse(model, batch):
    y = batch.in_times
    yT = batch.last_times
    k = batch.in_types
    model.to(y)
    x1 = model.inverse(y, k)
    x2, xT = model.inverse(y, k, yT)
    mask = k < model.n_marks
    assert torch.allclose(x1[mask], x2[mask], atol=1e-6)
    assert xT.shape == yT.shape


def test_model_same_y_yT(model, batch):
    # data
    y = batch.in_times
    B, _ = y.shape
    yT = batch.last_times
    k = batch.in_types

    # remove the last event from each sequence
    idxT = (k < model.n_marks).sum(dim=-1) - 1
    _k = k.clone()
    _k[torch.arange(B), idxT] = model.n_marks

    # index of mark last event
    idxk = k[torch.arange(B), idxT]

    # assert that the last event in y is equal to yT
    assert torch.allclose(y[torch.arange(B), idxT], yT[torch.arange(B), idxk])
    assert torch.all(yT[:, 0].unsqueeze(-1).expand(-1, model.n_marks) == yT)

    model.to(y)

    # assert that the last transformed time in x is equal to
    # the corresponding entry in _xT
    x = model.inverse(y, k)
    _, _xT = model.inverse(y, _k, yT)
    assert torch.allclose(x[torch.arange(B), idxT], _xT[torch.arange(B), idxk])
    # assert that the last transformed time in y is equal to
    # the corresponding entry in _yT
    yhat = model(x, k)
    _, _yhatT = model(x, _k, _xT)
    assert torch.allclose(yhat[torch.arange(B), idxT], _yhatT[torch.arange(B), idxk])

    # random time for last event; idea is to replicate sampling
    for _ in range(20):
        # sample last event
        k_rand = torch.randint(model.n_marks, (B,), device=k.device)
        k[torch.arange(B), idxT] = k_rand
        rand_interval = torch.rand((B,), device=y.device).exp()
        y[torch.arange(B), idxT] = y[torch.arange(B), idxT - 1] + rand_interval
        yT = y[torch.arange(B), idxT].unsqueeze(-1).expand(-1, model.n_marks)

        # index of mark last event
        idxk = k[torch.arange(B), idxT]

        # assert that the last event in y is equal to yT
        assert torch.allclose(y[torch.arange(B), idxT], yT[torch.arange(B), idxk])
        assert torch.all(yT[:, 0].unsqueeze(-1).expand(-1, model.n_marks) == yT)

        # assert that the last transformed time in x is equal to
        # the corresponding entry in _xT
        x = model.inverse(y, k)
        _, _xT = model.inverse(y, _k, yT)
        assert torch.allclose(x[torch.arange(B), idxT], _xT[torch.arange(B), idxk])
        # assert that the last transformed time in y is equal to
        # the corresponding entry in _yT
        yhat = model(x, k)
        _, _yhatT = model(x, _k, _xT)
        assert torch.allclose(
            yhat[torch.arange(B), idxT], _yhatT[torch.arange(B), idxk]
        )


def test_model_log_prob(model, batch):
    y = batch.in_times
    yT = batch.last_times
    k = batch.in_types
    model.to(y)
    log_prob = model.log_prob(y, k, yT)
    assert log_prob.shape[0] == y.shape[0]


def test_model_loss(model, batch):
    y = batch.in_times
    yT = batch.last_times
    k = batch.in_types
    model.to(y)
    loss = model.loss(y, k, yT)
    assert loss.is_floating_point()


def debug_sample_clamp(model, y, y_sample, k_sample, n_samples):
    total_samples = (k_sample[:, :, 1:] < model.n_marks).sum()
    faulty_ix = torch.where(
        (y_sample[:, :, 1:] < y[:, :-1])[k_sample[:, :, 1:] < model.n_marks]
    )
    faulty_sample = y_sample[:, :, 1:][k_sample[:, :, 1:] < model.n_marks][faulty_ix]
    faulty_y = y[:, :-1].expand(n_samples, -1, -1)[k_sample[:, :, 1:] < model.n_marks][
        faulty_ix
    ]
    faulty_n = faulty_sample.shape[0]
    if faulty_n > 0:
        faulty_diff = faulty_sample - faulty_y
        faulty_mean = faulty_diff.mean()
        faulty_min = faulty_diff.min()
        print(
            f"Faulty samples, n: {faulty_sample.shape[0]:,d} / {total_samples:,d}, %: {faulty_sample.shape[0] / total_samples:.4f}, mean: {faulty_mean:.4f}, min: {faulty_min:.4f}."
        )


def test_model_sample(model, batch):
    y = batch.in_times
    k = batch.in_types
    model.to(y)
    n_samples = 100
    y_sample, k_sample = model.sample(y, k, n_samples)
    assert y_sample.shape == (n_samples, y.shape[0], y.shape[1] + 1)
    assert k_sample.shape == (n_samples, k.shape[0], k.shape[1] + 1)
    # operation is broadcasted
    # k_sample.shape = (n_samples, B, N)
    assert torch.all((F.pad(k, (1, 0)) < model.n_marks) == (k_sample < model.n_marks))
    # test y_sample is bigger than previous y since this is a one-step ahead sample
    # not designed to catch every corner case since test will depend on model parameters
    # TODO: Due to clamping it is possible to sample the next event time before the previous event time
    # debug_sample_clamp(model, y, y_sample, k_sample, n_samples)
    assert torch.all((y_sample >= F.pad(y, (1, 0)))[k_sample < model.n_marks])


def test_model_expected_event(model, batch):
    y = batch.in_times
    k = batch.in_types
    t_max = 50
    model.to(y)
    n_samples = 2_000
    y_pred, k_pred = model.expected_event(y, k, t_max, n_samples)
    assert y_pred.shape == (y.shape[0], y.shape[1] + 1)
    assert k_pred.shape == (k.shape[0], k.shape[1] + 1)
    # ensure we only predict one-step ahead
    assert torch.all((F.pad(k, (1, 0)) < model.n_marks) == (k_pred < model.n_marks))
    # test y_pred is bigger than previous y since this is a one-step ahead sample
    assert torch.all((y_pred >= F.pad(y, (1, 0)))[k_pred < model.n_marks])


def test_model_predict_event_time(model, batch):
    y = batch.in_times
    k = batch.in_types
    model.to(y)
    n_samples = 100
    y_sample, _ = model.sample(y, k, n_samples)
    y_pred = model.predict_event_time(y_sample)
    assert y_pred.shape == (y.shape[0], y.shape[1] + 1)


def test_model_predict_event_prob(model, batch):
    y = batch.in_times
    k = batch.in_types
    model.to(y)
    n_samples = 100
    _, k_sample = model.sample(y, k, n_samples)
    k_prob = model.predict_event_prob(k_sample)
    assert k_prob.shape == (k.shape[0], k.shape[1] + 1, model.n_marks)
    assert torch.allclose(k_prob.sum(dim=-1) == 0, F.pad(k, (1, 0)) == model.n_marks)
    assert torch.all(
        (k_prob.sum(dim=-1)[F.pad(k, (1, 0)) < model.n_marks] - 1.0).abs() < 1e-6
    )


def test_model_expected_event_prob(model, batch):
    y = batch.in_times
    k = batch.in_types
    t_max = 50
    model.to(y)
    k_prob = model.expected_event_prob(y, k, t_max)
    assert k_prob.shape == (k.shape[0], k.shape[1] + 1, model.n_marks)
    assert torch.allclose(k_prob.sum(dim=-1) == 0, F.pad(k, (1, 0)) == model.n_marks)
    assert torch.all(
        (k_prob.sum(dim=-1)[F.pad(k, (1, 0)) < model.n_marks] - 1.0).abs() < 1e-6
    )


def test_model_predict_event_type(model, batch):
    y = batch.in_times
    k = batch.in_types
    idxT = (k < model.n_marks).sum(dim=-1) - 1
    model.to(y)
    t_max = 50
    k_prob = model.expected_event_prob(y, k, t_max)
    k_pred1 = model.predict_event_type(k_prob)
    idxT_pred1 = (k_pred1 < model.n_marks).sum(dim=-2) - 1
    assert torch.all((idxT.unsqueeze(-1) + 1) == idxT_pred1)
    assert k_pred1.shape == (k_prob.shape[0], k_prob.shape[1], 1)
    k_pred2 = model.predict_event_type(k_prob, model.n_marks)
    idxT_pred2 = (k_pred2 < model.n_marks).sum(dim=-2) - 1
    assert torch.all((idxT.unsqueeze(-1) + 1) == idxT_pred2)
    assert k_pred2.shape == (k_prob.shape[0], k_prob.shape[1], model.n_marks)


def test_model_cumulative_risk_func(model, batch):
    y = batch.in_times
    k = batch.in_types
    model.to(y)
    x = model.cumulative_risk_func(y, k)
    assert y.shape == x.shape


def test_model_intensity(model, batch):
    y = batch.in_times
    k = batch.in_types
    model.to(y)
    dt = 0.5
    t_max = 4
    yT, intensity = model.intensity_func(y, k, dt, t_max)
    assert yT.shape == (int(t_max / dt),)
    assert intensity.shape == (y.shape[0], int(t_max / dt), model.n_marks)
    assert torch.all(intensity >= 0)


def test_model_generate_cold_start(model, batch):
    y = torch.tensor([[], []], dtype=batch.in_times.dtype, device=batch.in_times.device)
    k = torch.tensor([[], []], dtype=batch.in_types.dtype, device=batch.in_types.device)
    model = model.to(y)
    n_samples = 10
    model.update_n_max(max(n_samples, batch.N_min))
    y_gen, k_gen = model.generate(y, k, n_samples=n_samples, N_min=batch.N_min)
    assert y_gen.shape == (y.shape[0], 10)
    assert k_gen.shape == (k.shape[0], 10)
    assert torch.all(y_gen > 0)
    assert torch.all(k_gen < model.n_marks)
    # test y_gen is bigger than previous y
    assert torch.all(y_gen[:, :-1] <= y_gen[:, 1:])


def test_model_generate_history(model, batch):
    y = batch.in_times
    k = batch.in_types
    model = model.to(y)
    n_samples = 10
    model.update_n_max(y.shape[1] + max(n_samples, batch.N_min))
    y_gen, k_gen = model.generate(y, k, n_samples=n_samples, N_min=batch.N_min)
    assert y_gen.shape == (y.shape[0], 10)
    assert k_gen.shape == (k.shape[0], 10)
    assert torch.all(y_gen > 0)
    assert torch.all(k_gen < model.n_marks)
    # test y_gen is bigger than previous y
    assert torch.all(y_gen[:, :-1] <= y_gen[:, 1:])

    idxT = (k < model.n_marks).sum(dim=-1)
    k[:, idxT.min() :] = model.n_marks
    y_gen, k_gen = model.generate(y, k, n_samples=n_samples, N_min=batch.N_min)
    assert y_gen.shape == (y.shape[0], 10)
    assert k_gen.shape == (k.shape[0], 10)
    assert torch.all(y_gen > 0)
    assert torch.all(k_gen < model.n_marks)
    # test y_gen is bigger than previous y
    assert torch.all(y_gen[:, :-1] <= y_gen[:, 1:])

    k[:, 0:] = model.n_marks
    y_gen, k_gen = model.generate(y, k, n_samples=n_samples, N_min=batch.N_min)
    assert y_gen.shape == (y.shape[0], 10)
    assert k_gen.shape == (k.shape[0], 10)
    assert torch.all(y_gen > 0)
    assert torch.all(k_gen < model.n_marks)
    # test y_gen is bigger than previous y
    assert torch.all(y_gen[:, :-1] <= y_gen[:, 1:])


# def test_model_forward_with_cache(model, batch):
#     # data
#     y = batch.in_times
#     _, N = y.shape
#     yT_vector = batch.last_times
#     yT_matrix = batch.in_times.unsqueeze(-1).expand(-1, -1, model.n_marks).unsqueeze(0)
#     k = batch.in_types
#     model.to(y)

#     x, xT_vector, cache = model.inverse(y, k, yT_vector, cache_y=True)
#     yhat1, yhat_vector1 = model(x, k, xT_vector)
#     yhat2, yhat_vector2 = model(x, k, xT_vector, cache=cache)

#     # TODO: due to clamping in the models, there will be small discrepancies in
#     # the results; it's difficult to create a mask that only targets
#     # non-clamped value
#     assert torch.allclose(y, yhat2)
#     if not hasattr(model, "clamps"):
#         assert torch.allclose(yhat1, yhat2, atol=1e-5)
#     assert torch.allclose(yhat_vector1, yhat_vector2, atol=1e-5)
