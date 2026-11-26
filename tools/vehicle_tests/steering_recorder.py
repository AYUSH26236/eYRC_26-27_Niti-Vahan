"""Observe steering across a physics step without changing controller commands."""
import csv


class SteeringRecorder:
    def __init__(self, sim, driver, handle):
        self.sim = sim
        self.joints = [sim.getObject(driver.STEER_LEFT_JOINT), sim.getObject(driver.STEER_RIGHT_JOINT)]
        self.writer = csv.writer(handle)
        self.writer.writerow(("t_command", "t_measured", "left_requested_rad", "right_requested_rad",
                              "left_actual_rad", "right_actual_rad"))

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def step(self):
        t = self.sim.getSimulationTime()
        requested = [self.sim.getJointTargetPosition(joint) for joint in self.joints]
        self.sim.step()
        measured = [self.sim.getJointPosition(joint) for joint in self.joints]
        self.writer.writerow([t, self.sim.getSimulationTime(), *requested, *measured])
