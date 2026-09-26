"""
analytics.py
------------
Turns raw rows from the database into the statistics and charts
required by the dashboard: hourly traffic, daily traffic, peak traffic,
average waiting (red-signal) time, and signal utilization, plus
Matplotlib bar/line/pie chart figures.
"""

from collections import defaultdict
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; Streamlit renders the returned Figure
import matplotlib.pyplot as plt
import pandas as pd

import config
from database import Database
from utils import get_logger

logger = get_logger(__name__)


class Analytics:
    def __init__(self, db: Database = None):
        self.db = db or Database()

    # ------------------------------------------------------------------
    def _traffic_dataframe(self, date: str = None) -> pd.DataFrame:
        rows = self.db.fetch_traffic_log(date)
        if not rows:
            return pd.DataFrame(columns=["date", "time", "road", "vehicle_count",
                                          "density", "signal_time", "emergency"])
        df = pd.DataFrame(rows)
        df["hour"] = df["time"].str.split(":").str[0]
        return df

    # ------------------------------------------------------------------
    def hourly_traffic(self, date: str = None) -> pd.DataFrame:
        df = self._traffic_dataframe(date)
        if df.empty:
            return df
        return df.groupby("hour")["vehicle_count"].sum().reset_index().rename(
            columns={"vehicle_count": "total_vehicles"}
        )

    def daily_traffic(self) -> pd.DataFrame:
        df = self._traffic_dataframe()
        if df.empty:
            return df
        return df.groupby("date")["vehicle_count"].sum().reset_index().rename(
            columns={"vehicle_count": "total_vehicles"}
        )

    def peak_traffic(self, date: str = None) -> dict:
        hourly = self.hourly_traffic(date)
        if hourly.empty:
            return {"hour": None, "total_vehicles": 0}
        peak_row = hourly.loc[hourly["total_vehicles"].idxmax()]
        return {"hour": peak_row["hour"], "total_vehicles": int(peak_row["total_vehicles"])}

    def average_green_time(self, date: str = None) -> float:
        df = self._traffic_dataframe(date)
        if df.empty:
            return 0.0
        return float(df["signal_time"].mean())

    def average_waiting_time(self, date: str = None) -> float:
        """
        Approximates average wait per road as: (sum of OTHER 3 roads' green
        times + their yellow/all-red overhead) for each cycle - i.e. how
        long a road's queue sits at red before its own green arrives.
        This is a simplified analytical estimate, not a per-vehicle
        measurement (that would require dwell-time tracking per vehicle).
        """
        df = self._traffic_dataframe(date)
        if df.empty:
            return 0.0
        overhead_per_road = config.YELLOW_TIME + config.ALL_RED_CLEARANCE
        avg_cycle_time = df["signal_time"].mean() + overhead_per_road
        num_other_roads = max(len(config.ROADS) - 1, 1)
        return round(avg_cycle_time * num_other_roads, 1)

    def signal_utilization(self, date: str = None) -> pd.DataFrame:
        """% of total green-time budget each road consumed."""
        df = self._traffic_dataframe(date)
        if df.empty:
            return df
        util = df.groupby("road")["signal_time"].sum().reset_index()
        total = util["signal_time"].sum()
        util["utilization_pct"] = (util["signal_time"] / total * 100).round(1) if total else 0
        return util

    def density_distribution(self, date: str = None) -> pd.DataFrame:
        df = self._traffic_dataframe(date)
        if df.empty:
            return df
        return df["density"].value_counts().rename_axis("density").reset_index(name="count")

    # ------------------------------------------------------------------
    # CHARTS (return matplotlib Figure objects; dashboard.py -> st.pyplot(fig))
    # ------------------------------------------------------------------
    def chart_hourly_bar(self, date: str = None):
        hourly = self.hourly_traffic(date)
        fig, ax = plt.subplots(figsize=(7, 4))
        if hourly.empty:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
            return fig
        ax.bar(hourly["hour"], hourly["total_vehicles"], color="#2E86AB")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Hourly Traffic Volume")
        fig.tight_layout()
        return fig

    def chart_daily_line(self):
        daily = self.daily_traffic()
        fig, ax = plt.subplots(figsize=(7, 4))
        if daily.empty:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
            return fig
        ax.plot(daily["date"], daily["total_vehicles"], marker="o", color="#F18F01")
        ax.set_xlabel("Date")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Daily Traffic Trend")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    def chart_density_pie(self, date: str = None):
        dist = self.density_distribution(date)
        fig, ax = plt.subplots(figsize=(5, 5))
        if dist.empty:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
            return fig
        colors = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#FF7043", "Very High": "#D32F2F"}
        ax.pie(
            dist["count"], labels=dist["density"], autopct="%1.1f%%",
            colors=[colors.get(d, "#999999") for d in dist["density"]],
        )
        ax.set_title("Traffic Density Distribution")
        return fig

    def chart_utilization_bar(self, date: str = None):
        util = self.signal_utilization(date)
        fig, ax = plt.subplots(figsize=(6, 4))
        if util.empty:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
            return fig
        ax.bar(util["road"], util["utilization_pct"], color="#6A4C93")
        ax.set_xlabel("Road")
        ax.set_ylabel("Green-Time Utilization (%)")
        ax.set_title("Signal Utilization by Road")
        fig.tight_layout()
        return fig
