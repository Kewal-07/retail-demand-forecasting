import numpy as np

from src.decision.simulate import simulate_inventory_policy


def _order_up_to_policy(demand: np.ndarray, target: float) -> np.ndarray:
    # order enough each day to bring inventory back up to target, so
    # leftover stock (which carries forward per the simulation's own
    # assumptions) reduces tomorrow's order instead of accumulating
    # unboundedly -- a fixed constant order every day regardless of
    # existing inventory is not a realistic policy under that assumption
    orders = np.zeros(len(demand))
    inventory = 0.0
    for t, d in enumerate(demand):
        orders[t] = max(target - inventory, 0)
        inventory += orders[t]
        inventory -= min(inventory, d)
    return orders


def test_policy_b_beats_policy_a_when_stockout_costly():
    rng = np.random.RandomState(42)
    demand = rng.poisson(10, 100).astype(float)

    orders_a = _order_up_to_policy(demand, target=10.0)  # policy A: point forecast (P50-like)
    orders_b = _order_up_to_policy(demand, target=15.0)  # policy B: newsvendor (P90-like)

    holding_cost, stockout_cost = 1.0, 9.0  # Cs much greater than Ch

    result_a = simulate_inventory_policy(demand, orders_a, holding_cost, stockout_cost)
    result_b = simulate_inventory_policy(demand, orders_b, holding_cost, stockout_cost)

    assert result_b["total_cost"] < result_a["total_cost"]


def test_fill_rate_bounded_0_1():
    rng = np.random.RandomState(0)
    demand = rng.poisson(10, 50).astype(float)
    orders = rng.poisson(10, 50).astype(float)

    result = simulate_inventory_policy(demand, orders, holding_cost=1.0, stockout_cost=5.0)

    assert 0.0 <= result["fill_rate"] <= 1.0
