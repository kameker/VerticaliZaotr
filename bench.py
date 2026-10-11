"""Test bench: runs the chair through the standard manoeuvres without a window and prints a report.

    python bench.py                         default chair, 80 kg / 1.75 m
    python bench.py --mass 150 --motor-amps 6.25 --battery-amps 14   heavier user, controller current limits
    python bench.py --only curb,turn        selected tests (see TESTS below)

The model is built into a temporary file next to this script; chair_human.xml is not touched.
"""
import argparse, math, os, pathlib, sys, tempfile, time
import numpy as np
import mujoco

import build_human as bh
import simulator as S

HERE = pathlib.Path(__file__).resolve().parent


def make(a, **over):
    kw = dict(mass=a.mass, height=a.height, act=a.act, motor=a.motor, motor_amps=a.motor_amps,
              battery_amps=a.battery_amps, chair=a.chair)
    kw.update(over)
    sim = S.Sim(kw.pop("mass"), kw.pop("height"), **kw)
    sim.last_input = float("inf")          # no panel: disable the dead-man timer
    return sim


def run(sim, seconds, chunk=0.05):
    t = sim.state
    for _ in range(max(1, int(round(seconds / chunk)))):
        t = sim.advance(chunk)
    return t


def body_x(sim, name):
    return float(sim.d.xpos[mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, name)][0])


def stand(sim):
    sim.cmd["lift"] = 1
    while sim.state["seat_angle"] < 74.5 and sim.d.time < 200:
        run(sim, 0.5)
    sim.cmd["lift"] = 0
    run(sim, 1.0)


# ------------------------------------------------------------------------------ tests
def t_loads(a):
    sim = make(a); t = run(sim, 1.5)
    rows = [("масса кресла / всего", f"{t['chair_mass']:.1f} / {t['total_mass']:.1f} кг"),
            ("нагрузка сидя: ведущее / колёсико", f"{t['load_drive_left']:.0f} / {t['load_front_left']:.0f} Н")]
    stand(sim); t = sim.state
    rows.append(("нагрузка стоя: ведущее / колёсико", f"{t['load_drive_left']:.0f} / {t['load_front_left']:.0f} Н"))
    return rows


def t_lift(a):
    sim = make(a); run(sim, 1.0)
    sim.cmd["lift"] = 1
    peak, t0 = 0.0, sim.d.time
    while sim.state["seat_angle"] < 74.5 and sim.d.time - t0 < 200:
        t = run(sim, 0.1); peak = max(peak, abs(t["lift_force"]))
    ok = sim.state["seat_angle"] >= 74.5
    return [("подъём до 75°", f"{sim.d.time - t0:.0f} с, пик {peak:.0f} Н" if ok else f"НЕ ВСТАЛ: {sim.state['seat_angle']:.0f}°")]


def t_speed(a):
    sim = make(a); sim.reset(y=-8.0, yaw=math.pi / 2); run(sim, 1.0)
    sim.cmd.update(level=5, jy=1.0)
    v = [run(sim, 0.1)["speed"] for _ in range(60)]
    amps = np.mean([run(sim, 0.1)["battery_A"] for _ in range(10)])
    v5 = max(v)
    return [("макс. скорость", f"{v5:.2f} м/с ({v5 * 3.6:.1f} км/ч)"),
            ("ток батареи на 5-й по ровному", f"{amps:.1f} А → запас хода ≈ {S.BATTERY_AH / amps * v5 * 3.6:.0f} км")]


def t_turn(a):
    sim = make(a); sim.reset(y=-5.0); run(sim, 1.0)
    p0 = sim.d.xpos[sim.frame].copy()
    sim.cmd.update(level=2, jx=1.0)
    r = [abs(run(sim, 0.1)["yaw_rate"]) for _ in range(50)]
    drift = float(np.linalg.norm((sim.d.xpos[sim.frame] - p0)[:2]))
    cmd = math.degrees(abs(sim.w_cmd))
    return [("разворот на месте (2-я)", f"{np.mean(r[25:]):.0f}°/с из {cmd:.0f}°/с, смещение центра {drift * 100:.0f} см")]


def climb(a, where, edge, h, level, jy=1.0, secs=15, standstill=False):
    """Drive at a step of height h with its front edge at x = edge; True if the drive axle got over it."""
    sim = make(a); run(sim, 1.0)
    x, z = bh.TRACK_SPOTS[where]
    if standstill:                         # casters touching the edge: no run-up
        r = float(sim.m.geom_size[mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_GEOM, "front_left_contact")][0])
        x = edge - math.sqrt(r * r - (r - h) ** 2) - (body_x(sim, "front_left") - body_x(sim, "frame")) - 0.005
    sim.teleport(x, z=z); run(sim, 1.0)
    sim.cmd.update(level=level, jy=jy)
    warns = set()
    for _ in range(int(secs / 0.1)):
        warns.update(run(sim, 0.1)["warnings"])
        if body_x(sim, "frame") > edge + 0.15:
            return True, warns
    return False, warns


def t_obstacles(a):
    rows = []
    for name, where, edge, h in (("порог 30 мм", "threshold", 2.47, 0.03), ("бордюр 50 мм", "curb50", bh.CURB_50_X, 0.05)):
        ok0, w0 = climb(a, where, edge, h, level=1, jy=0.4, standstill=True)
        res = [f"с места: {'да' if ok0 else 'нет'}"]
        need = next((lv for lv in range(1, 6) if climb(a, where, edge, h, level=lv)[0]), None)
        res.append(f"с разгона 1,5 м: {'с ' + str(need) + '-й передачи' if need else 'нет'}")
        rows.append((name, ", ".join(res) + (f" [{sorted(w0)[0]}]" if w0 and not ok0 else "")))
    return rows


class Amps:
    """Peak battery and motor currents over a run."""
    def __init__(self):
        self.bat = self.mot = 0.0

    def add(self, t):
        self.bat = max(self.bat, t["battery_A"])
        self.mot = max(self.mot, t["motor_A_left"], t["motor_A_right"])

    def text(self):
        return f"; ток батареи до {self.bat:.1f} А, мотора до {self.mot:.1f} А"


def level_speed(a, level):
    return min(bh.motor_profile(a.motor)["v_rated"], S.TOP_SPEED_CAP) * S.LEVEL_FRACTIONS[level - 1]


def t_slopes(a):
    rows = []
    v2 = level_speed(a, 2)
    sim = make(a); run(sim, 1.0)
    sim.event({"name": "goto", "where": "ramp"}); run(sim, 1.0)
    sim.cmd.update(level=2, jy=1.0)
    on_ramp, amps = [], Amps()
    for _ in range(100):
        t = run(sim, 0.1); amps.add(t)
        if 5.2 < body_x(sim, "frame") < 6.8: on_ramp.append(t["speed"])
    rows.append((f"пандус 1:12 вверх (2-я, {v2:.2f} м/с)", (f"скорость на подъёме {min(on_ramp):.2f}…{max(on_ramp):.2f} м/с"
                 if on_ramp else "не доехал") + amps.text()))
    sim = make(a); run(sim, 1.0)
    sim.event({"name": "goto", "where": "platform"}); run(sim, 0.5)
    sim.cmd.update(level=2, jy=1.0)
    vmax = max(run(sim, 0.1)["speed"] for _ in range(60))
    rows.append((f"спуск 1:8 (2-я, цель {v2:.2f} м/с)", f"макс. {vmax:.2f} м/с"))
    for deg in bh.HILLS:
        start, run_, rise = bh.HILL_LAYOUT[deg]
        sim = make(a); run(sim, 1.0)
        sim.event({"name": "goto", "where": f"hill{deg}"}); run(sim, 1.0)
        sim.cmd.update(level=2, jy=1.0)
        warns, top, amps = set(), False, Amps()
        for _ in range(300):
            t = run(sim, 0.1); warns.update(t["warnings"]); amps.add(t)
            if body_x(sim, "frame") > start + run_ + 0.2:
                top = True; break
            if abs(t["pitch"]) > 40: break
        z = float(sim.d.xpos[sim.frame][2])
        res = ("въехал" if top else f"не въехал (поднялся на {z:.2f} из {rise:.2f} м)") + amps.text()
        bad = [w for w in sorted(warns) if "упёрлось" in w or "ОПРОКИН" in w or "оторвалось" in w]
        rows.append((f"уклон {deg}° вверх (2-я)", res + (f" [{bad[0]}]" if bad else "")))
    return rows


def t_descent(a):
    """Long slope (tilted gravity), 1st gear downhill forwards and backwards for 3 s, then let go of the joystick."""
    rows, v1 = [], level_speed(a, 1)
    for deg in (6, 10, 15):
        res = []
        for name, sign, jy in (("вперёд", 1, 1.0), ("задом", -1, -1.0)):
            sim = make(a); sim.reset(y=-5.0); run(sim, 1.0)
            sim.cmd["slope_fwd"] = sign * deg; run(sim, 1.0)
            sim.cmd.update(level=1, jy=jy); run(sim, 3.0)
            v = abs(sim.state["speed"])
            sim.cmd.update(jy=0.0)
            over = any(abs(run(sim, 0.03)["pitch"]) > 45 for _ in range(100))
            held = v < v1 + 0.15
            res.append(f"{name} {v:.2f} м/с" + ("" if held else " — разгоняется") + (", отпустил: ОПРОКИНУЛОСЬ" if over else ""))
        rows.append((f"спуск {deg}° на 1-й (задано {v1:.2f} м/с)", "; ".join(res)))
    return rows


def t_park(a):
    rows = []
    for direction, key, sign in (("носом вниз", "slope_fwd", 1), ("носом вверх", "slope_fwd", -1), ("боком", "slope_side", 1)):
        held = 0.0
        for deg in (3, 6, 8, 10, 12, 15, 20):
            sim = make(a); sim.reset(y=-5.0); run(sim, 1.0)
            sim.cmd["park"] = True; sim.cmd[key] = sign * deg; run(sim, 0.5)
            p0 = sim.d.xpos[sim.frame].copy(); run(sim, 3.0)
            if np.linalg.norm((sim.d.xpos[sim.frame] - p0)[:2]) > 0.02 or abs(sim.state["pitch"]) > 20 or abs(sim.state["roll"]) > 20:
                break
            held = deg
        rows.append((f"стояночный тормоз, {direction}", f"держит до {held:g}°"))
    return rows


def t_tip(a):
    rows = []
    for standing in (False, True):
        for name, key, sign in (("вперёд", "slope_fwd", 1), ("назад", "slope_fwd", -1), ("вбок", "slope_side", 1)):
            sim = make(a); sim.reset(y=-5.0); run(sim, 1.0)
            if standing: stand(sim)
            sim.cmd["park"] = True; run(sim, 0.3)
            first, tip = None, None
            for deg in np.arange(1, 45.1, 0.5):
                sim.cmd[key] = sign * deg; t = run(sim, 0.4)
                if first is None and t["lifted"]: first = deg
                if abs(t["pitch"]) > 20 or abs(t["roll"]) > 20: tip = deg; break
            rows.append((f"{'стоя' if standing else 'сидя'}: наклон пола {name}",
                         f"колесо оторвалось при {first:g}°, опрокинулось при {tip:g}°" if tip else
                         f"колесо оторвалось при {first}°, не опрокинулось до 45°"))
    return rows


TESTS = dict(loads=t_loads, lift=t_lift, speed=t_speed, turn=t_turn, curb=t_obstacles, slope=t_slopes,
             descent=t_descent, park=t_park, tip=t_tip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=80.0)
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--act", default=bh.DEFAULT_ACTUATOR, choices=list(bh.ACTUATORS))
    ap.add_argument("--motor", default=bh.DEFAULT_MOTOR, choices=[k for k in bh.MOTORS if k != "custom"])
    ap.add_argument("--motor-amps", type=float, default=None, help="controller limit per motor, A (default: rated)")
    ap.add_argument("--battery-amps", type=float, default=bh.DEFAULT_BATTERY_AMPS, help="battery current limit, A")
    ap.add_argument("--chair", default=bh.DEFAULT_CHAIR, choices=list(bh.CHAIRS))
    ap.add_argument("--only", default=",".join(TESTS))
    a = ap.parse_args()
    fd, tmp = tempfile.mkstemp(suffix=".xml", prefix=".bench_", dir=HERE)   # next to meshes/ for the CAD chair
    os.close(fd)
    S.MODEL = pathlib.Path(tmp)
    try:
        print(f"{bh.CHAIRS[a.chair]} · {a.mass:g} кг / {a.height:g} м · {bh.ACTUATORS[a.act]['name']} · "
              f"{bh.MOTORS[a.motor]['name']} · ток мотора ≤{a.motor_amps or bh.MOTORS[a.motor]['I']:g} А, "
              f"батареи ≤{a.battery_amps:g} А")
        for name in a.only.split(","):
            t0 = time.time()
            for k, v in TESTS[name](a):
                print(f"  {k:38s} {v}")
            sys.stdout.flush()
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    main()
