import numpy as np

from src.models.reconcile import build_summing_matrix, mint_reconcile


def test_reconciled_forecasts_sum_correctly():
    hierarchy_map = {
        "Total": ["Store_A", "Store_B"],
        "Store_A": ["Item_1", "Item_2"],
        "Store_B": ["Item_3"],
    }
    S = build_summing_matrix(hierarchy_map)

    # key order matches build_summing_matrix's node order: hierarchy_map
    # keys first (Total, Store_A, Store_B), then leaves (Item_1, Item_2, Item_3)
    point_forecasts = {
        "Total": np.array([100.0, 110.0]),
        "Store_A": np.array([55.0, 60.0]),
        "Store_B": np.array([50.0, 55.0]),  # intentionally incoherent inputs
        "Item_1": np.array([20.0, 22.0]),
        "Item_2": np.array([30.0, 33.0]),
        "Item_3": np.array([45.0, 50.0]),
    }

    reconciled = mint_reconcile(point_forecasts, S)

    np.testing.assert_allclose(
        reconciled["Item_1"] + reconciled["Item_2"], reconciled["Store_A"]
    )
    np.testing.assert_allclose(
        reconciled["Store_A"] + reconciled["Store_B"], reconciled["Total"]
    )


def test_reconciliation_preserves_shape():
    hierarchy_map = {"Total": ["A", "B"]}
    S = build_summing_matrix(hierarchy_map)
    point_forecasts = {
        "Total": np.array([10.0, 20.0, 30.0]),
        "A": np.array([4.0, 9.0, 14.0]),
        "B": np.array([5.0, 10.0, 15.0]),
    }

    reconciled = mint_reconcile(point_forecasts, S)

    assert set(reconciled.keys()) == set(point_forecasts.keys())
    for key in point_forecasts:
        assert len(reconciled[key]) == len(point_forecasts[key])
