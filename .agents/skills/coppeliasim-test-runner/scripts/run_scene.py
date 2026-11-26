"""Launch one isolated CoppeliaSim run using a project's scenario adapter (POSIX)."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid


def read_adapter(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("vehicle_scenario", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def worker(adapter, scene, out, port, token):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    task = read_adapter(Path(adapter))
    client = RemoteAPIClient(host="127.0.0.1", port=int(port))
    # A request queued during simulator startup must have time to receive a reply.
    client.initialTimeout = 60
    sim = client.require("sim")
    owner = sim.getNamedStringParam("vehicleTestOwner")
    if isinstance(owner, bytes):
        owner = owner.decode()
    if owner != token:
        raise RuntimeError("Test instance identity mismatch")
    sim.loadScene(scene)
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("Expected a newly loaded, stopped scene")
    sim.setStepping(True)
    vehicle = sim.getObject(task.VEHICLE)
    start = sim.getObjectPose(vehicle, sim.handle_world)
    if sim.getSimulationTime() != 0:
        raise RuntimeError("Scene did not start at simulation time zero")
    folder = Path(out)
    save(folder / "start.json", dict(vehicle=task.VEHICLE, pose_world=start,
                                     time_s=0, timestep_s=sim.getSimulationTimeStep(),
                                     simulator_version=sim.getInt32Param(sim.intparam_program_version)))
    task.run(sim, folder)
    checks = task.evaluate(folder)
    if not isinstance(checks, list) or not checks or any(
            not isinstance(c, dict) or not c.get("name") or type(c.get("passed")) is not bool
            for c in checks):
        raise ValueError("Expected a nonempty list of named boolean checks")
    save(folder / "checks.json", checks)
    return 0 if all(c["passed"] for c in checks) else 1


def terminate(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()


def run(args, out):
    task = read_adapter(args.adapter)
    original = (args.adapter.parent / task.SCENE).resolve(strict=True)
    if original.suffix.lower() != ".ttt":
        raise ValueError("Expected a .ttt scene")
    scene = out / original.name
    shutil.copy2(original, scene)
    save(out / "inputs.json", dict(source=str(original), working_copy=scene.name,
                                   scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest()))
    if hasattr(task, "stage_assets"):
        task.stage_assets(original.parent, out)
    executable = os.getenv("COPPELIASIM_BIN") or shutil.which("coppeliaSim") or shutil.which("coppeliaSim.sh")
    if not executable and sys.platform == "darwin":
        executable = "/Applications/coppeliasim.app/Contents/MacOS/coppeliaSim"
    if not executable:
        raise RuntimeError("Set COPPELIASIM_BIN to the simulator executable")
    token = uuid.uuid4().hex
    port = 49152 + secrets.randbelow(16384)
    command = [executable, "-h", "-vscriptinfos", "-GzmqRemoteApi.rpcPort=" + str(port),
               "-GvehicleTestOwner=" + token]
    server = driver = None
    try:
        with (out / "simulator.log").open("w") as simlog, (out / "controller.log").open("w") as runlog:
            server = subprocess.Popen(command, stdout=simlog, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            driver = subprocess.Popen([sys.executable, "-B", "-u", __file__, "--worker",
                                       str(args.adapter), str(scene), str(out), str(port), token],
                                      cwd=out, stdout=runlog, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            deadline = time.monotonic() + args.timeout
            while driver.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"CoppeliaSim exited during the run ({server.returncode}). See simulator.log")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Exceeded {args.timeout}s wall time. See simulator.log and controller.log")
                time.sleep(0.1)
            if driver.returncode not in (0, 1) or not (out / "checks.json").exists():
                details = (out / "controller.log").read_text(errors="replace").splitlines()
                raise RuntimeError(details[-1] if details else "Controller worker exited without checks")
            return driver.returncode
    finally:
        terminate(driver)
        terminate(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vehicle-test-runs"))
    parser.add_argument("--timeout", type=int, default=300, metavar="SECONDS")
    args = parser.parse_args()
    args.adapter = args.adapter.resolve()
    args.output.mkdir(exist_ok=True, parents=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=args.output)).resolve()
    print(f"Run directory: {out}", flush=True)
    try:
        code = run(args, out)
        report = dict(status="PASS" if code == 0 else "FAIL", checks=json.loads((out / "checks.json").read_text()))
    except (Exception, KeyboardInterrupt) as exc:
        code = 2
        report = dict(status="ERROR", reason=str(exc) or type(exc).__name__)
    save(out / "result.json", report)
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        try:
            sys.exit(worker(*sys.argv[2:]))
        except Exception:
            traceback.print_exc()
            sys.exit(2)
    sys.exit(main())
