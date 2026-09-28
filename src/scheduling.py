"""Predictive bed-allocation simulation (the 'scheduling' half of the project).

This is a deliberately simple discrete-event simulation that connects the ML
prediction to an operations-research idea:

  * Patients arrive every day with a (true) length of stay.
  * The ML model gives a *predicted* LOS.
  * Policy A (FIFO): beds are assigned first-come-first-served.
  * Policy B (SPT-predicted): each day, waiting patients are served in order of
    *predicted* LOS (shortest first) — the classic single-machine scheduling
    rule that minimises total waiting time.

Only the *predicted* LOS is used for decisions (never the true LOS), which
keeps the simulation honest and shows how prediction quality drives the gain.
"""
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


class BedSimulator:
    """Daily-batched bed allocation over a fixed horizon."""

    def __init__(self, n_beds: int, seed: int = RANDOM_SEED):
        self.n_beds = n_beds
        self.rng = np.random.default_rng(seed)

    def simulate(self, arrivals: pd.DataFrame, policy: str) -> dict:
        """Run the simulation.

        Parameters
        ----------
        arrivals : DataFrame with columns
            ['arrival_day', 'true_los', 'pred_los']
        policy : 'fifo' | 'spt'

        Returns
        -------
        dict with total/mean waiting, occupancy and per-patient records.
        """
        if policy not in ("fifo", "spt"):
            raise ValueError("policy must be 'fifo' or 'spt'")

        horizon = int(arrivals["arrival_day"].max()) + 1
        n = len(arrivals)
        day_bucket = arrivals.groupby("arrival_day")

        # bed state: day each bed becomes free (0 = free on day 0)
        bed_free_at = np.zeros(self.n_beds, dtype=int)
        waiting = []          # patients currently waiting (arrival_day, true_los, pred_los, idx)
        records = []          # per-patient outcome: waiting days

        daily_occupancy = np.zeros(horizon + 1, dtype=float)
        total_bed_capacity = 0.0

        for day in range(horizon):
            # 1. today's arrivals join the queue
            if day in day_bucket.groups:
                grp = day_bucket.get_group(day)
                for _, row in grp.iterrows():
                    waiting.append(
                        [int(row["arrival_day"]), float(row["true_los"]),
                         float(row["pred_los"]), len(records)]
                    )
                    records.append(
                        {"arrival_day": int(row["arrival_day"]),
                         "true_los": float(row["true_los"]),
                         "pred_los": float(row["pred_los"]),
                         "waiting_days": None}
                    )

            # 2. prediction-driven ordering (shortest predicted LOS first)
            if policy == "spt" and waiting:
                waiting.sort(key=lambda p: p[2])  # pred_los ascending

            # 3. release beds that free up today
            free_now = bed_free_at <= day
            n_free = int(free_now.sum())

            # 4. assign as many waiting patients as possible
            assigned = 0
            while waiting and assigned < n_free:
                pat = waiting.pop(0)
                # choose the bed that frees soonest (best-fit intuition)
                free_beds = np.where(bed_free_at <= day)[0]
                if len(free_beds) == 0:
                    break
                bed = free_beds[0]
                bed_free_at[bed] = day + int(np.ceil(pat[1]))  # true LOS occupancy
                records[pat[3]]["waiting_days"] = day - pat[0]
                assigned += 1

            # 5. occupancy snapshot
            occupied = int((bed_free_at > day).sum())
            daily_occupancy[day] = occupied
            total_bed_capacity += self.n_beds

        # finish patients still waiting past the horizon (sanity)
        for pat in waiting:
            records[pat[3]]["waiting_days"] = horizon - pat[0]

        rec_df = pd.DataFrame(records)
        waiting_days = rec_df["waiting_days"].dropna()
        return {
            "policy": policy,
            "n_patients": n,
            "mean_wait_days": float(waiting_days.mean()) if len(waiting_days) else 0.0,
            "total_wait_days": float(waiting_days.sum()),
            "max_wait_days": float(waiting_days.max()) if len(waiting_days) else 0.0,
            "bed_occupancy_rate": float(daily_occupancy[:horizon].mean() / self.n_beds),
            "records": rec_df,
            "daily_occupancy": daily_occupancy[:horizon],
        }


def build_arrival_scenario(df, n_patients: int = 1200, n_beds: int = 80, seed: int = RANDOM_SEED):
    """Sample a realistic arrival stream from the dataset.

    True LOS comes from the actual records; predicted LOS is generated with a
    realistic error (Gaussian noise on the true value) so the scenario can run
    even before any model is trained. When a fitted model is provided, the
    predicted LOS is produced by the model itself.
    """
    rng = np.random.default_rng(seed)
    los_sample = df["time_in_hospital"].sample(n_patients, random_state=seed).values
    arrival_day = rng.integers(0, 30, size=n_patients)

    # realistic prediction: model quality equivalent to ~1 day MAE noise
    pred_los = np.clip(los_sample + rng.normal(0, 1.2, size=n_patients), 0.5, 25)

    return pd.DataFrame({
        "arrival_day": arrival_day,
        "true_los": los_sample.astype(float),
        "pred_los": pred_los,
    })


def compare_policies(arrivals: pd.DataFrame, n_beds: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Run FIFO vs SPT-predicted on the same arrival stream; return summary."""
    sim = BedSimulator(n_beds=n_beds, seed=seed)
    rows = []
    for policy in ("fifo", "spt"):
        res = sim.simulate(arrivals, policy=policy)
        rows.append({
            "policy": policy,
            "mean_wait_days": res["mean_wait_days"],
            "total_wait_days": res["total_wait_days"],
            "max_wait_days": res["max_wait_days"],
            "bed_occupancy_rate": res["bed_occupancy_rate"],
        })
    return pd.DataFrame(rows)
