import pytest

import torch
from torch.testing import make_tensor

import multittpp.flows
from multittpp.flows.transformer import (
    CausalSelfAttention,
    TransformerMLP,
    AutoregressiveTransformer,
)


def flowid(param):
    if isinstance(param, tuple):
        name = str(param[0].__name__)
        if len(param[1]) > 0 or len(param[2]) > 0:
            name += "("
        if len(param[1]) > 0:
            name += str(param[1])[1:-1] + ("" if len(param[2]) > 0 else ", ")
        if len(param[2]) > 0:
            name += str(param[2])[1:-1].replace("'", "").replace(": ", "=")
        if len(param[1]) > 0 or len(param[2]) > 0:
            name += ")"
        return name
    else:
        return str(param.__name__) + "()"


def flow_fixture(request, device, shape, n_marks):
    if isinstance(request.param, tuple):
        Flow = request.param[0]
        _args = request.param[1]
        _kwargs = request.param[2].copy()
        _kwargs["n_marks"] = n_marks
        if "scale_init" in _kwargs and isinstance(_kwargs["scale_init"], torch.Tensor):
            _kwargs["scale_init"] = _kwargs["scale_init"][:n_marks]
        if "shift_init" in _kwargs and isinstance(_kwargs["shift_init"], torch.Tensor):
            _kwargs["shift_init"] = _kwargs["shift_init"][:n_marks]
        if "n_events" in _kwargs:
            _kwargs["n_events"] = shape[-2]
        out = Flow(*_args, **_kwargs)
    else:
        out = request.param(n_marks=n_marks)
    if isinstance(out, multittpp.flows.BlockDiagonal):
        with torch.no_grad():
            out.params[:] = torch.rand_like(out.params).abs()
    out.to(device)
    return out


@pytest.fixture(
    params=[
        #
        # INVERTIBLE FLOWS
        #
        multittpp.flows.Affine,
        (multittpp.flows.Affine, [], {"shift_init": 2.0}),
        (multittpp.flows.Affine, [], {"scale_init": 2.0}),
        (
            multittpp.flows.Affine,
            [],
            {
                "shift_init": torch.tensor([-1.0, -2.0]),
                "scale_init": torch.tensor([3.0, 2.0]),
            },
        ),
        (
            multittpp.flows.BlockDiagonal,
            [],
            {"block_size": 5, "offset": 1},
        ),
        multittpp.flows.CumSum,
        (multittpp.flows.Spline, [], {"n_knots": 20}),
        (
            multittpp.flows.SplineTransformer,
            [],
            {
                "n_events": None,
                "n_embd": 10,
                "n_heads": 2,
                "n_blocks": 4,
                "n_knots": 20,
                "dropout": 0,
            },
        ),
        # SEMI-INVERTIBLE FLOWS
        #
        # Clamped flows are not completely invertible. We need to clamp the
        # input to check whether invertibility holds.
        multittpp.flows.Exp,  # forward clamp
        multittpp.flows.ExpNegative,  # forward clamp
        multittpp.flows.Sigmoid,  # forward clamp
        multittpp.flows.Log,  # inverse clamp
        multittpp.flows.NegativeLog,  # inverse clamp
        multittpp.flows.Logit,  # inverse clamp
    ],
    ids=flowid,
)
def flow(request, device, shape, n_marks):
    return flow_fixture(request, device, shape, n_marks)


def exponential_noise(batch, flow):
    # x as draws from the exponential distribution
    x = torch.rand(
        (batch.in_times.shape[0], batch.in_times.shape[1]), device=batch.in_times.device
    ).exp()
    k = batch.in_types
    idxT = (k < flow.n_marks).sum(dim=-1) - 1
    xT = torch.empty(
        (x.shape[0], flow.n_marks), device=batch.in_times.device, dtype=x.dtype
    )
    for mark in range(flow.n_marks):
        idxT = torch.argmax(
            (mark == k) * torch.arange(x.shape[1], device=k.device), dim=-1
        )
        xT[:, mark] = x[torch.arange(x.shape[0]), idxT]
        # adjust when mark does not show up in the sequence
        not_in_sequence = ~torch.any(mark == k, dim=-1)
        xT[not_in_sequence, mark] = torch.rand(
            (not_in_sequence.sum(),), device=xT.device
        ).exp()

    return x, k, xT


def assert_invertible(input, k, inputT, flow, direction, test_no_inputT=False):
    if direction == "forward":
        reverse_direction = "inverse"
        f = flow
        f_inverse = flow.inverse
    elif direction == "inverse":
        reverse_direction = "forward"
        f = flow.inverse
        f_inverse = flow
    else:
        raise ValueError("direction must be either 'forward' or 'inverse'")

    # assert we are not producing nans and getting correct shapes
    assert not input.isnan().any()
    assert not inputT.isnan().any()

    out, log_det_jac, outT, log_det_jacT = f(input, k, inputT)
    assert out.shape == input.shape
    assert log_det_jac.shape == input.shape
    assert outT.shape == inputT.shape
    assert log_det_jacT.shape == inputT.shape
    assert not out.isnan().any()
    assert not log_det_jac.isnan().any()
    assert not outT.isnan().any()
    assert not log_det_jacT.isnan().any()

    input_hat, inv_log_det_jac, input_hatT, inv_log_det_jacT = f_inverse(out, k, outT)
    assert input_hat.shape == out.shape
    assert inv_log_det_jac.shape == out.shape
    assert input_hatT.shape == outT.shape
    assert inv_log_det_jacT.shape == outT.shape
    assert not input_hat.isnan().any()
    assert not inv_log_det_jac.isnan().any()
    assert not input_hatT.isnan().any()
    assert not inv_log_det_jacT.isnan().any()

    # mask
    mask = k < flow.n_marks
    if inputT.dim() < 4:
        maskT = torch.ones_like(inputT, dtype=bool)
    else:
        n_samples, B, N, _ = inputT.shape
        maskT = mask.unsqueeze(-1).expand(n_samples, B, N, flow.n_marks)
    if hasattr(flow, "clamps"):
        if flow.clamp_direction == direction:
            mask = mask & (out > flow.clamps[0]) & (out < flow.clamps[1])
            maskT = maskT & (outT > flow.clamps[0]) & (outT < flow.clamps[1])

    # tests
    if hasattr(flow, "clamps") and flow.clamp_direction == reverse_direction:
        assert torch.allclose(
            input.clamp(flow.clamps[0], flow.clamps[1])[mask],
            input_hat[mask],
            atol=1e-6,
        )

        mask = mask & (input > flow.clamps[0]) & (input < flow.clamps[1])
        assert torch.allclose(log_det_jac[mask], -inv_log_det_jac[mask], atol=1e-5)

        assert torch.allclose(
            inputT.clamp(flow.clamps[0], flow.clamps[1])[maskT],
            input_hatT[maskT],
            atol=1e-6,
        )

        maskT = maskT & (inputT > flow.clamps[0]) & (inputT < flow.clamps[1])
        assert torch.allclose(log_det_jacT[maskT], -inv_log_det_jacT[maskT], atol=1e-5)

    else:
        assert torch.allclose(input[mask], input_hat[mask], atol=1e-6)
        assert torch.allclose(log_det_jac[mask], -inv_log_det_jac[mask], atol=1e-5)

        assert torch.allclose(inputT[maskT], input_hatT[maskT], atol=1e-6)
        assert torch.allclose(log_det_jacT[maskT], -inv_log_det_jacT[maskT], atol=1e-5)

    if test_no_inputT:
        out, log_det_jac = f(input, k)
        input_hat, inv_log_det_jac = f_inverse(out, k)

        mask = k < flow.n_marks
        if hasattr(flow, "clamps"):
            if flow.clamp_direction == direction:
                mask = mask & (out > flow.clamps[0]) & (out < flow.clamps[1])

        if (
            hasattr(flow, "clamp_direction")
            and flow.clamp_direction == reverse_direction
        ):
            assert torch.allclose(
                input.clamp(flow.clamps[0], flow.clamps[1])[mask],
                input_hat[mask],
                atol=1e-6,
            )

            mask = mask & (input > flow.clamps[0]) & (input < flow.clamps[1])
            assert torch.allclose(log_det_jac[mask], -inv_log_det_jac[mask], atol=1e-5)
        else:
            assert torch.allclose(input[mask], input_hat[mask], atol=1e-6)
            assert torch.allclose(log_det_jac[mask], -inv_log_det_jac[mask], atol=1e-5)


def test_flow_clamp_attr(flow):
    if hasattr(flow, "clamps"):
        assert hasattr(flow, "clamp_direction")
        assert flow.clamp_direction in ["forward", "inverse"]


def test_flow_n_marks(flow, n_marks):
    assert flow.n_marks == n_marks


def test_flow_forward_invertible(batch, flow):
    x, k, xT = exponential_noise(batch, flow)
    assert_invertible(
        input=x,
        k=k,
        inputT=xT,
        flow=flow,
        direction="forward",
        test_no_inputT=True,
    )


def test_flow_inverse_invertible(batch, flow):
    y = batch.in_times
    yT = batch.last_times
    k = batch.in_types
    assert_invertible(
        input=y,
        k=k,
        inputT=yT,
        flow=flow,
        direction="inverse",
        test_no_inputT=True,
    )


def test_flow_multiT_forward_invertible_v1(batch, flow):
    """
    Test whether the flow is invertible when the last event vector xT is shaped
    as (n_samples, B, K).
    """
    x, k, xT = exponential_noise(batch, flow)
    xT = torch.stack((xT, 2 * xT))
    assert_invertible(
        input=x,
        k=k,
        inputT=xT,
        flow=flow,
        direction="forward",
        test_no_inputT=False,
    )


def test_flow_multiT_inverse_invertible_v1(batch, flow):
    """
    Test whether the flow is invertible when the last event vector yT is shaped
    as (n_samples, B, K).
    """
    # data
    y = batch.in_times
    yT = batch.last_times
    yT = torch.stack((yT, 2 * yT))
    k = batch.in_types
    assert_invertible(
        input=y,
        k=k,
        inputT=yT,
        flow=flow,
        direction="inverse",
        test_no_inputT=False,
    )


def test_flow_multiT_forward_invertible_v2(batch, flow):
    """
    Test whether the flow is invertible when the last event vector xT is shaped
    as (n_samples, B, N, K).
    """
    x, k, _ = exponential_noise(batch, flow)
    xT = x.unsqueeze(-1).expand(-1, -1, flow.n_marks)
    xT = torch.stack((xT, 2 * xT))
    assert_invertible(
        input=x,
        k=k,
        inputT=xT,
        flow=flow,
        direction="forward",
        test_no_inputT=False,
    )


def test_flow_multiT_inverse_invertible_v2(batch, flow):
    """
    Test whether the flow is invertible when the last event vector yT is shaped
    as (n_samples, B, N, K).
    """
    y = batch.in_times
    yT = batch.in_times.unsqueeze(-1).expand(-1, -1, flow.n_marks)
    yT = torch.stack((yT, 2 * yT))
    k = batch.in_types
    assert_invertible(
        input=y,
        k=k,
        inputT=yT,
        flow=flow,
        direction="inverse",
        test_no_inputT=False,
    )


def test_flow_same_y_yT(batch, flow):
    """
    Test whether the inverse and forward pass returns the same value whether
    we apply it to the history or to the next event.

    A flow applied to a vector of event time y and marks k returns a
    vector with transformed time x.

    The same flow applied to the vector of next candidate times yT with history
    y and _k where the last event has been remove from _k should return a
    vector with transformed proposed times _xT such that the last index in x
    should equal to the equivalent entry in _xT.
    """
    # data
    y = batch.in_times
    B, _ = y.shape
    yT = batch.last_times
    k = batch.in_types

    # remove the last event from each sequence
    idxT = (k < flow.n_marks).sum(dim=-1) - 1
    _k = k.clone()
    _k[torch.arange(B), idxT] = flow.n_marks

    # index of mark last event
    idxk = k[torch.arange(B), idxT]

    # assert that the last event in y is equal to yT
    assert torch.allclose(y[torch.arange(B), idxT], yT[torch.arange(B), idxk])
    assert torch.all(yT[:, 0].unsqueeze(-1).expand(-1, flow.n_marks) == yT)

    # assert that the last transformed time in x is equal to
    # the corresponding entry in _xT
    x, log_det_jac = flow.inverse(y, k)
    _, _, _xT, _log_det_jacT = flow.inverse(y, _k, yT)
    assert torch.allclose(x[torch.arange(B), idxT], _xT[torch.arange(B), idxk])
    assert torch.allclose(
        log_det_jac[torch.arange(B), idxT],
        _log_det_jacT[torch.arange(B), idxk],
        atol=1e-5,
    )
    # assert that the last transformed time in y is equal to
    # the corresponding entry in _yT
    yhat, inv_log_det_jac = flow(x, k)
    _, _, _yhatT, _inv_log_det_jacT = flow(x, _k, _xT)
    assert torch.allclose(yhat[torch.arange(B), idxT], _yhatT[torch.arange(B), idxk])
    assert torch.allclose(
        inv_log_det_jac[torch.arange(B), idxT],
        _inv_log_det_jacT[torch.arange(B), idxk],
        atol=1e-5,
    )

    # random time for last event; idea is to replicate sampling
    for _ in range(20):
        # sample last event
        k_rand = torch.randint(flow.n_marks, (B,), device=k.device)
        k[torch.arange(B), idxT] = k_rand
        rand_interval = torch.rand((B,), device=y.device).exp()
        y[torch.arange(B), idxT] = y[torch.arange(B), idxT - 1] + rand_interval
        yT = y[torch.arange(B), idxT].unsqueeze(-1).expand(-1, flow.n_marks)

        # index of mark last event
        idxk = k[torch.arange(B), idxT]

        # assert that the last event in y is equal to yT
        assert torch.allclose(y[torch.arange(B), idxT], yT[torch.arange(B), idxk])
        assert torch.all(yT[:, 0].unsqueeze(-1).expand(-1, flow.n_marks) == yT)

        # assert that the last transformed time in x is equal to
        # the corresponding entry in _xT
        x, log_det_jac = flow.inverse(y, k)
        _, _, _xT, _log_det_jacT = flow.inverse(y, _k, yT)
        assert torch.allclose(x[torch.arange(B), idxT], _xT[torch.arange(B), idxk])
        assert torch.allclose(
            log_det_jac[torch.arange(B), idxT],
            _log_det_jacT[torch.arange(B), idxk],
            atol=1e-5,
        )
        # assert that the last transformed time in y is equal to
        # the corresponding entry in _yT
        yhat, inv_log_det_jac = flow(x, k)
        _, _, _yhatT, _inv_log_det_jacT = flow(x, _k, _xT)
        assert torch.allclose(
            yhat[torch.arange(B), idxT], _yhatT[torch.arange(B), idxk]
        )
        assert torch.allclose(
            inv_log_det_jac[torch.arange(B), idxT],
            _inv_log_det_jacT[torch.arange(B), idxk],
            atol=1e-5,
        )


def test_flow_same_y_yT_multi(batch, flow):
    """
    Test whether the inverse and forward pass returns the same value whether
    we apply it to the history or to the next event.

    A flow applied to a vector of event time y and marks k returns a
    vector with transformed time x.

    The same flow applied to the matrix of next candidate times yT with history
    y and k should return a matrix with transformed proposed times xT which equal x.
    """
    # data
    y = batch.in_times
    B, _ = y.shape
    yT_matrix = batch.in_times.unsqueeze(-1).expand(-1, -1, flow.n_marks).unsqueeze(0)
    k = batch.in_types

    x, log_det_jac, xT_matrix, log_det_jacT_matrix = flow.inverse(y, k, yT_matrix)

    _k = k.masked_fill(k == flow.n_marks, 0).unsqueeze(-1)
    _x = xT_matrix[0].gather(-1, _k).squeeze(-1)
    _log_det_jac = log_det_jacT_matrix[0].gather(-1, _k).squeeze(-1)

    assert torch.allclose(x[k < flow.n_marks], _x[k < flow.n_marks])
    assert torch.allclose(log_det_jac[k < flow.n_marks], _log_det_jac[k < flow.n_marks])

    yhat, inv_log_det_jac, yhatT_matrix, inv_log_det_jacT_matrix = flow.inverse(
        x, k, xT_matrix
    )

    _yhat = yhatT_matrix[0].gather(-1, _k).squeeze(-1)
    _inv_log_det_jac = inv_log_det_jacT_matrix[0].gather(-1, _k).squeeze(-1)

    assert torch.allclose(yhat[k < flow.n_marks], _yhat[k < flow.n_marks])
    assert torch.allclose(
        inv_log_det_jac[k < flow.n_marks], _inv_log_det_jac[k < flow.n_marks]
    )


def test_flow_same_yT_yT_multi(batch, flow):
    """
    Test whether parallel and serial computation of yT produce the same result.
    """
    # data
    y = batch.in_times
    _, N = y.shape
    yT_matrix = batch.in_times.unsqueeze(-1).expand(-1, -1, flow.n_marks).unsqueeze(0)
    k = batch.in_types

    x, _, xT_matrix, log_det_jacT_matrix = flow.inverse(y, k, yT_matrix)
    _, _, yhatT_matrix, inv_log_det_jacT_matrix = flow(x, k, xT_matrix)

    for i in range(N):
        _k = k.clone()
        _k[:, i:] = flow.n_marks
        mask = _k[:, i] < flow.n_marks

        yT_vector = y[:, [i]] * torch.ones(flow.n_marks, dtype=y.dtype, device=y.device)

        x, _, xT_vector, log_det_jacT_vector = flow.inverse(y, _k, yT_vector)

        assert torch.allclose(xT_matrix[0, mask, i, :], xT_vector[mask])
        assert torch.allclose(
            log_det_jacT_matrix[0, mask, i, :], log_det_jacT_vector[mask], atol=1e-5
        )

        _, _, yhatT_vector, inv_log_det_jacT_vector = flow(x, _k, xT_vector)

        assert torch.allclose(yhatT_matrix[0, mask, i, :], yhatT_vector[mask])
        assert torch.allclose(
            inv_log_det_jacT_matrix[0, mask, i, :],
            inv_log_det_jacT_vector[mask],
            atol=1e-5,
        )


def test_flow_forward_with_y(batch, flow):
    # data
    y = batch.in_times
    _, N = y.shape
    yT_vector = batch.last_times
    yT_matrix = batch.in_times.unsqueeze(-1).expand(-1, -1, flow.n_marks).unsqueeze(0)
    k = batch.in_types

    x, _, xT_vector, _ = flow.inverse(y, k, yT_vector)
    yhat1, _, yhat_vector1, inv_log_det_jacT_vector1 = flow(x, k, xT_vector)
    yhat2, _, yhat_vector2, inv_log_det_jacT_vector2 = flow(x, k, xT_vector, y)

    assert torch.allclose(y, yhat2)
    if not hasattr(flow, "clamps"):
        assert torch.allclose(yhat1, yhat2, atol=1e-6)
    assert torch.allclose(yhat_vector1, yhat_vector2)
    assert torch.allclose(inv_log_det_jacT_vector1, inv_log_det_jacT_vector2)

    x, _, xT_matrix, _ = flow.inverse(y, k, yT_matrix)
    yhat1, _, yhat_matrix1, inv_log_det_jacT_matrix1 = flow(x, k, xT_matrix)
    yhat2, _, yhat_matrix2, inv_log_det_jacT_matrix2 = flow(x, k, xT_matrix, y)

    assert torch.allclose(y, yhat2)
    if not hasattr(flow, "clamps"):
        assert torch.allclose(yhat1, yhat2, atol=1e-6)
    # TODO: test tends to be slightly unstable for BlockDiagonal
    assert torch.allclose(yhat_matrix1, yhat_matrix2, atol=1e-6)
    assert torch.allclose(inv_log_det_jacT_matrix1, inv_log_det_jacT_matrix2, atol=1e-6)
