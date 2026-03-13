import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from abc import ABC, abstractmethod
from scipy.spatial import distance
import pandas as pd

# ── Load Track from CSV ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "small_track.csv"))

BLUE_CONES   = df[df["tag"] == "blue"      ][["x", "y"]].values.astype(float)
YELLOW_CONES = df[df["tag"] == "yellow"    ][["x", "y"]].values.astype(float)
BIG_ORANGE   = df[df["tag"] == "big_orange"][["x", "y"]].values.astype(float)

_cs               = df[df["tag"] == "car_start"].iloc[0]
CAR_START_POS     = np.array([float(_cs["x"]), float(_cs["y"])])
CAR_START_HEADING = float(_cs["direction"])   # radians (0 = east)

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])


# ── Build Approximate Centerline ──────────────────────────────────────────────
def _build_centerline():
    """
    Pair each blue cone with its nearest yellow cone, take the midpoint,
    then sort CLOCKWISE around the track centroid so pure-pursuit drives CW.
    """
    center = np.mean(MAP_CONES, axis=0)
    D      = distance.cdist(BLUE_CONES, YELLOW_CONES)
    mids   = np.array(
        [(BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
         for i in range(len(BLUE_CONES))]
    )
    angles = np.arctan2(mids[:, 1] - center[1], mids[:, 0] - center[0])
    return mids[np.argsort(angles)[::-1]]   # descending angle = clockwise


CENTERLINE = _build_centerline()


# ── Simulation Parameters ─────────────────────────────────────────────────────
SENSOR_RANGE = 12.0   # metres – sensor visibility radius
NOISE_STD    = 0.20   # metres – measurement noise std-dev
WHEELBASE    = 3.0    # metres – bicycle model wheelbase
DT           = 0.1    # seconds – time step
SPEED        = 7.0    # m/s
LOOKAHEAD    = 5.5    # pure-pursuit lookahead distance (m)
N_FRAMES     = 130    # ≈ one full lap


# ── Utility Functions ─────────────────────────────────────────────────────────
def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def pure_pursuit(pos: np.ndarray, heading: float, path: np.ndarray) -> float:
    """Compute steering angle (rad) to follow *path* via pure-pursuit."""
    dists   = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))
    n       = len(path)
    target  = path[(nearest + 5) % n]       # fallback lookahead
    for k in range(nearest, nearest + n):
        pt = path[k % n]
        if np.linalg.norm(pt - pos) >= LOOKAHEAD:
            target = pt
            break
    alpha = angle_wrap(
        np.arctan2(target[1] - pos[1], target[0] - pos[0]) - heading
    )
    steer = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), LOOKAHEAD)
    return float(np.clip(steer, -0.6, 0.6))


def local_to_global(local_pts: np.ndarray,
                    pos: np.ndarray, heading: float) -> np.ndarray:
    """Rotate + translate points from the car's local frame to world frame."""
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, -s], [s, c]])       # local → world rotation
    return (R @ local_pts.T).T + pos


def get_measurements(pos: np.ndarray, heading: float) -> np.ndarray:
    """
    Simulate a 2-D lidar: return visible cone positions as noisy
    measurements in the car's LOCAL frame (x = forward, y = left).
    """
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible = MAP_CONES[dists < SENSOR_RANGE]
    if len(visible) == 0:
        return np.zeros((0, 2))
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])       # world → local (transpose of above)
    local = (R @ (visible - pos).T).T
    return local + np.random.normal(0, NOISE_STD, local.shape)


def draw_track(ax, alpha_b: float = 0.4, alpha_y: float = 0.4) -> None:
    ax.scatter(BLUE_CONES[:, 0],   BLUE_CONES[:, 1],
               c="royalblue", marker="^", s=65,  alpha=alpha_b,
               zorder=2, label="Blue cones")
    ax.scatter(YELLOW_CONES[:, 0], YELLOW_CONES[:, 1],
               c="gold",      marker="^", s=65,  alpha=alpha_y,
               zorder=2, label="Yellow cones")
    ax.scatter(BIG_ORANGE[:, 0],   BIG_ORANGE[:, 1],
               c="darkorange", marker="s", s=100, alpha=0.7,
               zorder=2, label="Start gate")


def draw_car(ax, pos: np.ndarray, heading: float) -> None:
    ax.scatter(pos[0], pos[1], c="red", s=160, zorder=7, label="Car")
    ax.arrow(pos[0], pos[1],
             2.2 * np.cos(heading), 2.2 * np.sin(heading),
             head_width=0.8, fc="red", ec="red", zorder=8)


def setup_ax(ax, subtitle: str = "") -> None:
    ax.set_xlim(-28, 28)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    if subtitle:
        ax.set_title(subtitle, fontsize=10)


# ── Abstract Base ─────────────────────────────────────────────────────────────
class Bot(ABC):
    def __init__(self):
        self.pos     = CAR_START_POS.copy()   # (2,) float64
        self.heading = CAR_START_HEADING      # radians

    @abstractmethod
    def data_association(self, measurements, current_map): ...

    @abstractmethod
    def localization(self, velocity, steering): ...

    @abstractmethod
    def mapping(self, measurements): ...


# ──  Solution ──────────────────────────────────────────────────────────
class Solution(Bot):
    def __init__(self):
        super().__init__()
        self.learned_map  = []                    # list of np.ndarray (2,)
        # Internal state exposed for visualisation
        self._global_meas = np.zeros((0, 2))
        self._assoc       = np.array([], dtype=int)

    # ------------------------------------------------------------------
    def data_association(self, measurements, current_map):
        """
        Nearest-Neighbour data association.
        Steps:
          1. Transform local measurements → world frame using current pose.
          2. For each measurement find the nearest cone in *current_map*.
        Returns an int array of indices into current_map.
        """
        if len(measurements) == 0 or len(current_map) == 0:
            self._global_meas = np.zeros((0, 2))
            self._assoc       = np.array([], dtype=int)
            return self._assoc

        gm = local_to_global(measurements, self.pos, self.heading)
        self._global_meas = gm

        D           = distance.cdist(gm, current_map)
        self._assoc = np.argmin(D, axis=1)
        return self._assoc

    # ------------------------------------------------------------------
    def localization(self, velocity, steering):
        """
        Bicycle kinematic model (dead reckoning):
            ẋ = v·cos(ψ)
            ẏ = v·sin(ψ)
            ψ̇ = (v / L)·tan(δ)
        """
        self.pos[0]  += velocity * np.cos(self.heading) * DT
        self.pos[1]  += velocity * np.sin(self.heading) * DT
        self.heading  = angle_wrap(
            self.heading + (velocity / WHEELBASE) * np.tan(steering) * DT
        )

    # ------------------------------------------------------------------
    def mapping(self, measurements):
        """
        Transform local measurements to world frame and accumulate unique
        landmark estimates (distance-threshold deduplication).
        """
        if len(measurements) == 0:
            return
        gm = local_to_global(measurements, self.pos, self.heading)
        for p in gm:
            if not self.learned_map or \
               min(np.linalg.norm(p - q) for q in self.learned_map) > 2.0:
                self.learned_map.append(p.copy())


# ── Problem 1 – Data Association ──────────────────────────────────────────────
def make_problem1():
    """
    Visualise nearest-neighbour association: cyan dots = sensor measurements
    transformed to world frame; green dashed lines connect each measurement
    to its matched map cone.
    """
    sol = Solution()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 1 – Data Association  (Nearest Neighbour)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas  = get_measurements(sol.pos, sol.heading)
        sol.data_association(meas, MAP_CONES)
        sol.localization(SPEED, steer)

        draw_track(ax)

        if len(sol._global_meas) > 0:
            for idx, gm in zip(sol._assoc, sol._global_meas):
                mc = MAP_CONES[idx]
                ax.plot([gm[0], mc[0]], [gm[1], mc[1]],
                        "g--", lw=1.0, alpha=0.65, zorder=3)
            ax.scatter(sol._global_meas[:, 0], sol._global_meas[:, 1],
                       c="cyan", s=45, zorder=5,
                       label=f"Measurements ({len(sol._global_meas)})")

        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax, f"Frame {frame+1}/{N_FRAMES}  –  "
                     "green lines = NN association")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


# ── Problem 2 – Localization ───────────────────────────────────────────────────
def make_problem2():
    """
    Visualise dead-reckoning: the magenta trail is the car's estimated
    trajectory built purely from the kinematic model and steering commands.
    """
    sol     = Solution()
    path_x  = [float(sol.pos[0])]
    path_y  = [float(sol.pos[1])]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 2 – Localization  (Dead Reckoning / Kinematic Model)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        sol.localization(SPEED, steer)
        path_x.append(float(sol.pos[0]))
        path_y.append(float(sol.pos[1]))

        draw_track(ax)
        ax.plot(path_x, path_y, color="magenta", lw=2.0,
                alpha=0.85, zorder=4, label="Dead-reckoning path")
        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"pos=({sol.pos[0]:.1f}, {sol.pos[1]:.1f})  "
            f"ψ={np.degrees(sol.heading):.1f}°")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


# ── Problem 3 – Mapping ───────────────────────────────────────────────────────
def make_problem3():
    """
    Visualise incremental mapping: green × marks show the car's accumulated
    global cone map built from local sensor measurements.  Ground-truth cones
    are faded so the learned map stands out.
    """
    sol = Solution()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 3 – Mapping  (Local → Global Transform + Deduplication)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas  = get_measurements(sol.pos, sol.heading)
        sol.localization(SPEED, steer)
        sol.mapping(meas)

        draw_track(ax, alpha_b=0.15, alpha_y=0.15)

        if sol.learned_map:
            lm = np.array(sol.learned_map)
            ax.scatter(lm[:, 0], lm[:, 1],
                       c="limegreen", marker="x", s=90, linewidths=2.0,
                       zorder=5, label=f"Mapped cones ({len(lm)})")

        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"map size: {len(sol.learned_map)} / {len(MAP_CONES)} cones")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Driverless Car Hackathon – SLAM Visualisation ===")
    print(f"  Blue cones   : {len(BLUE_CONES)}")
    print(f"  Yellow cones : {len(YELLOW_CONES)}")
    print(f"  Big orange   : {len(BIG_ORANGE)}")
    print(f"  Car start    : {CAR_START_POS}  "
          f"heading={np.degrees(CAR_START_HEADING):.1f}°")
    print(f"  Centerline   : {len(CENTERLINE)} waypoints (clockwise)")
    print("\nOpening 3 animation windows …")

    # Keep references to prevent garbage collection of FuncAnimation objects.
    fig1, ani1 = make_problem1()
    fig2, ani2 = make_problem2()
    fig3, ani3 = make_problem3()

    plt.show()