from rydstate import BasisMQDT


def test_mqdt_basis_creation() -> None:
    """Smoke-test that the basis builds and respects the nu range."""
    basis = BasisMQDT("Sr88_mqdt", nu=(25, 30))
    assert len(basis) > 0
    assert all(25 <= s.nu_ref <= 30 for s in basis.states)


def test_mqdt_basis_coefficients_normalized() -> None:
    """Every state must have unit-norm coefficients."""
    basis = BasisMQDT("Sr88_mqdt", nu=(25, 30))
    for state in basis.states:
        assert abs(state.norm - 1) < 1e-6


def test_mqdt_basis_f_tot_filter() -> None:
    """When f_tot is passed, only states with that total angular momentum are returned."""
    basis = BasisMQDT("Sr88_mqdt", nu=(25, 30), f_tot=0.0)
    assert len(basis) > 0
    for state in basis.states:
        assert abs(state.angular.calc_exp_qn("f_tot") - 0.0) < 1e-10


def test_mqdt_basis_sort_and_filter() -> None:
    """sort_states and filter_states work on MQDT states."""
    basis = BasisMQDT("Sr88_mqdt", nu=(25, 30))
    basis.sort_states("nu_ref")
    nu_refs = [s.nu_ref for s in basis.states]
    assert nu_refs == sorted(nu_refs)

    n_before = len(basis)
    basis.filter_states("nu_ref", (27.0, 28.0))
    assert len(basis) < n_before
    assert all(27.0 - 1e-10 <= s.nu_ref <= 28.0 + 1e-10 for s in basis.states)
