# parameters.py

class SimulationParameters:
    def __init__(self, RN=1.0, RT=1.0, tau=0.1, t_span=(0, 100), t_points=1000, nonlinear=False):
        # Physical parameters
        self.RN = RN
        self.RT = RT
        self.tau = tau

        # Simulation parameters
        self.nonlinear = nonlinear