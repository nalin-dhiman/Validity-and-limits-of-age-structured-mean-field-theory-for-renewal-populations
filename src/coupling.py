class MeanFieldCoupling:
    def __init__(self, J, tau_s, dt):
        self.J = J
        self.tau_s = tau_s
        self.dt = dt
        self.x = 0.0

    def step(self, A_prev):
        """
        Update synaptic variable x.
        tau_s * dx/dt = -x + A
        We define the Output Coupling = J * x
        """
        dx = (-self.x + A_prev) * (self.dt / self.tau_s)
        self.x += dx
        return self.x # Return raw state, access output property for J*x

    @property
    def output(self):
        return self.J * self.x

    def reset(self):
        self.x = 0.0
