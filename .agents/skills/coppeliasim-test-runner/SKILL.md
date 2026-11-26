---
name: coppeliasim-test-runner
description: Run a CoppeliaSim vehicle controller on a fresh copy of a .ttt scene, capture its trajectory, and check task-specific criteria after controller edits.
---

# CoppeliaSim test runner

Reuse the project's test command when present. In the Niti Vahan Task 1B workspace, run `./test-vehicle` from the directory containing that launcher. It reads the current saved controller on every invocation. Read the run's `result.json` and use the measurements for the failed segment to diagnose the next edit.

## Bind another scene

1. Identify the exact scene, vehicle body, controller entry point, simulator version, and task criteria from the project. Prefer its documented input sequence and official evaluator when available. Label local tolerances explicitly. Keep controller implementation, submission files and reference trajectories intact when adding the harness.
2. Use the bundled [run_scene.py](scripts/run_scene.py) for macOS or Linux with CoppeliaSim's ZeroMQ remote API and a compatible Python environment. Set `COPPELIASIM_BIN` if the executable is outside the default lookup. Reuse the project's native launcher on other platforms.
3. Write a small Python adapter defining `SCENE` (an absolute path or a path relative to the adapter's directory), `VEHICLE` (absolute scene object path), `run(sim, folder)`, and `evaluate(folder)`. Resolve relative paths with `Path(__file__).resolve().parent`. The worker loads a new scene copy, records the vehicle's world pose at time zero, and enables stepping. If the scene reads sibling assets, define optional `stage_assets(source_dir, folder)` to copy those into the same layout beside the working scene before it loads. Seed randomized inputs where the controller or scene exposes a seed.
4. In `run`, invoke the real controller in its documented mode. Pass the provided simulator connection. Give exactly one component ownership of simulation steps. For an embedded controller, start the simulator, sample the world pose at time zero and after each step, advance to a bounded simulation time and stop. For an external controller with an existing stepping loop, call that loop and preserve its CSV. Capture the exact inputs and record time, world x and y in meters. Add heading, contact, stop or checkpoint events when the rubric requires them.
5. In `evaluate`, check completeness and finite monotonic timestamps before judging the path. Return a nonempty list of dictionaries containing `name`, boolean `passed`, and relevant measurements with units. Compare checkpoint order or lane occupancy over the required intervals. A correct endpoint alone cannot confirm the route. Record missing evaluator-only criteria in the summary.
6. Put one launcher in the project that chooses the correct Python environment and calls `python /path/to/run_scene.py /path/to/scenario.py --output /path/to/runs`. Reuse this exact command after each edit. Each invocation creates its own output folder. Exit 0 means the measured criteria passed, 1 means a criterion failed, and 2 means the run could not be evaluated.

Check the start pose, input hashes and simulation settings when comparing runs. Point to the path trace and per-criterion result. If the simulator cannot launch, report the specific startup error and treat archived or synthetic traces as offline checks only.
