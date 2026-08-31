"""
Dashboard - generates the two artifacts your requirements doc says to lead
the demo with: the coverage matrix (breadth) and the detection-rate-by-round
chart (learning happened). Static matplotlib output, no server needed - fine
for a hackathon demo, runs directly in a Colab cell.

Usage (Colab):
    from dashboard.build_dashboard import build_dashboard
    build_dashboard()   # reads data/coverage_matrix.csv + data/dashboard_log.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_coverage_matrix(coverage_path: str = "data/coverage_matrix.csv", ax=None):
    df = pd.read_csv(coverage_path)
    if df.empty:
        print("Coverage matrix is empty - run at least one round first.")
        return

    counts = df.groupby("manipulation_type").size().sort_values(ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_title(f"Scenario coverage by category ({len(df)} total scenarios)")
    ax.set_ylabel("number of scenarios")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.05, str(v), ha="center")
    return ax


def print_coverage_table(coverage_path: str = "data/coverage_matrix.csv"):
    df = pd.read_csv(coverage_path)
    if df.empty:
        print("Coverage matrix is empty.")
        return df
    display_cols = ["scenario_name", "f3_tactic", "f3_technique", "manipulation_type",
                     "novelty_tag", "detection_rate", "round_added"]
    return df[display_cols]


def plot_detection_rate_over_rounds(dashboard_log_path: str = "data/dashboard_log.csv", ax=None):
    df = pd.read_csv(dashboard_log_path)
    if df.empty:
        print("Dashboard log is empty - run at least one round first.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["round"], df["ensemble_recall"], marker="o", label="recall")
    ax.plot(df["round"], df["ensemble_precision"], marker="s", label="precision")
    ax.plot(df["round"], df["ensemble_f1"], marker="^", label="F1")
    ax.plot(df["round"], df["blue_fpr"], marker="x", label="FPR", linestyle="--", color="red")
    ax.set_xlabel("round")
    ax.set_ylabel("score")
    ax.set_title("Detection efficacy over rounds (never show recall alone - FPR included)")
    ax.legend()
    ax.set_ylim(-0.02, 1.02)
    return ax


def plot_reward_over_rounds(dashboard_log_path: str = "data/dashboard_log.csv", ax=None):
    df = pd.read_csv(dashboard_log_path)
    if df.empty:
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["round"], df["blue_reward"], marker="o", label="blue_reward")
    ax.plot(df["round"], df["mean_red_reward"], marker="s", label="mean_red_reward")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("reward")
    ax.set_title("Red vs Blue reward over rounds (self-play signal)")
    ax.legend()
    return ax


def per_scenario_detection_heatmap(coverage_path: str = "data/coverage_matrix.csv", ax=None):
    """
    A per-scenario-type detection-rate bar chart - answers the requirements
    doc's "break efficacy down per scenario type, not just aggregate" point.
    """
    df = pd.read_csv(coverage_path)
    df = df[df["detection_rate"].notna()]
    if df.empty:
        print("No scored scenarios yet.")
        return

    df = df.sort_values("detection_rate")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(df))))
    colors = ["#C44E52" if v < 0.5 else "#55A868" for v in df["detection_rate"]]
    ax.barh(df["scenario_name"], df["detection_rate"], color=colors)
    ax.set_xlabel("detection rate")
    ax.set_title("Per-scenario detection rate (red = still missed, green = caught)")
    ax.set_xlim(0, 1)
    return ax


def build_dashboard(
    coverage_path: str = "data/coverage_matrix.csv",
    dashboard_log_path: str = "data/dashboard_log.csv",
    save_path: str = None,
):
    """Builds all 4 panels in one figure - call this once at the end of your run."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    plot_coverage_matrix(coverage_path, ax=axes[0, 0])
    plot_detection_rate_over_rounds(dashboard_log_path, ax=axes[0, 1])
    plot_reward_over_rounds(dashboard_log_path, ax=axes[1, 0])
    per_scenario_detection_heatmap(coverage_path, ax=axes[1, 1])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")
    plt.show()

    print("\n=== Coverage matrix table ===")
    table = print_coverage_table(coverage_path)
    if table is not None:
        print(table.to_string(index=False))


if __name__ == "__main__":
    # Smoke test with fabricated data (no LLM/API key needed) so this file
    # can be verified standalone.
    import os
    os.makedirs("data", exist_ok=True)

    coverage_demo = pd.DataFrame([
        {"scenario_id": "s1", "scenario_name": "low-and-slow", "f3_tactic": "Evasion",
         "f3_technique": "Low-and-Slow Velocity Abuse", "manipulation_type": "behavioral",
         "fields_manipulated": "amount|mean_inter_txn_seconds", "novelty_tag": "baseline",
         "round_added": 0, "detection_rate": 0.85, "red_reward": 0.3, "times_missed": 1},
        {"scenario_id": "s2", "scenario_name": "device ring", "f3_tactic": "Monetization",
         "f3_technique": "Mule Network Cash-Out", "manipulation_type": "network",
         "fields_manipulated": "device_fingerprint|ip_address_hash", "novelty_tag": "baseline",
         "round_added": 0, "detection_rate": 0.40, "red_reward": 0.9, "times_missed": 3},
        {"scenario_id": "s3", "scenario_name": "card testing", "f3_tactic": "Monetization",
         "f3_technique": "Card-Not-Present Abuse", "manipulation_type": "channel",
         "fields_manipulated": "amount|channel", "novelty_tag": "baseline",
         "round_added": 1, "detection_rate": 0.72, "red_reward": 0.5, "times_missed": 1},
    ])
    coverage_demo.to_csv("data/coverage_matrix_demo.csv", index=False)

    log_demo = pd.DataFrame([
        {"round": 0, "ensemble_recall": 0.55, "ensemble_precision": 0.60, "ensemble_f1": 0.57,
         "blue_fpr": 0.08, "blue_reward": 0.20, "mean_red_reward": 0.65},
        {"round": 1, "ensemble_recall": 0.68, "ensemble_precision": 0.71, "ensemble_f1": 0.69,
         "blue_fpr": 0.05, "blue_reward": 0.35, "mean_red_reward": 0.55},
        {"round": 2, "ensemble_recall": 0.79, "ensemble_precision": 0.75, "ensemble_f1": 0.77,
         "blue_fpr": 0.03, "blue_reward": 0.50, "mean_red_reward": 0.45},
    ])
    log_demo.to_csv("data/dashboard_log_demo.csv", index=False)

    build_dashboard(
        coverage_path="data/coverage_matrix_demo.csv",
        dashboard_log_path="data/dashboard_log_demo.csv",
        save_path="data/dashboard_demo.png",
    )

    os.remove("data/coverage_matrix_demo.csv")
    os.remove("data/dashboard_log_demo.csv")
    print("\nDemo smoke test complete (demo CSVs cleaned up, PNG kept).")
