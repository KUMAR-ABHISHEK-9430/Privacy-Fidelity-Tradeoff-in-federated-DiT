from opacus.accountants import RDPAccountant
from fdit.private import calibrate, privacy_record


def test_monotonic_persistent_and_calibrated():
    q, steps, delta = .02, 12, 1e-6
    sigma = calibrate(3., delta, q, steps)
    acc = RDPAccountant()
    previous = 0.
    for round_id in range(3):
        for _ in range(4):
            acc.step(noise_multiplier=sigma, sample_rate=q)
        eps = acc.get_epsilon(delta)
        assert eps > previous
        previous = eps
        new = RDPAccountant()
        new.load_state_dict(acc.state_dict())
        assert new.get_epsilon(delta) == eps
        acc = new
    assert sum(h[2] for h in acc.history) == steps
    assert acc.get_epsilon(delta) <= 3.02
    config = {"delta": delta, "expected_logical_batch_size": 2, "max_grad_norm": 1.}
    record = privacy_record([acc] * 3, [steps] * 3, [sigma] * 3, [q] * 3, config, 3.)
    assert record["epsilon"] == acc.get_epsilon(delta)
