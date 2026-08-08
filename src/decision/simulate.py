"""Inventory simulation. See CLAUDE.md Section 6D and Section 9.

Stated assumptions: no lead time (an order placed for day t arrives on
day t), lost sales not backordered (unmet demand is lost, not carried
into future demand), leftover stock carries forward (unsold inventory
rolls into the next day's starting inventory).
"""
import numpy as np


def simulate_inventory_policy(
    demand: np.ndarray, orders: np.ndarray, holding_cost: float, stockout_cost: float
) -> dict:
    demand = np.asarray(demand, dtype="float64")
    orders = np.asarray(orders, dtype="float64")

    inventory = 0.0
    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    total_fulfilled = 0.0
    total_demand = 0.0
    stockout_days = 0
    inventory_levels = []

    for d, o in zip(demand, orders):
        inventory += o  # no lead time
        fulfilled = min(inventory, d)
        unmet = d - fulfilled  # lost, not backordered
        inventory -= fulfilled  # leftover carries forward

        total_holding_cost += inventory * holding_cost
        total_stockout_cost += unmet * stockout_cost
        total_fulfilled += fulfilled
        total_demand += d
        if unmet > 0:
            stockout_days += 1
        inventory_levels.append(inventory)

    total_cost = total_holding_cost + total_stockout_cost
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 1.0

    return {
        "total_cost": float(total_cost),
        "fill_rate": float(fill_rate),
        "stockout_days": stockout_days,
        "avg_inventory": float(np.mean(inventory_levels)) if inventory_levels else 0.0,
    }
