
class PIDController():
    """
    PID controller

    """

    def __init__(self, Kp: float, Ki: float, Kd: float, filter_amount: float):
        """
        Initialization of controller

        Params:
            Kp: proportional gain
            Ki: integrator gain
            Kd: derivative gain
            filter_amount: how much the error will be filtered before taking the derivative. 1 = no filter and 0 will make error not change

        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.filter_amount = filter_amount
        self.past_error = None
        self.I = 0

    def __call__(self, error: float):
        # Proportional
        P = error * self.Kp
        # Integral
        self.I = error * self.Ki + self.I
        # Derivative
        if self.past_error is None: self.past_error = error
        filtered_error = (1 - self.filter_amount) * error + self.filter_amount * self.past_error
        D = self.Kd * (filtered_error - self.past_error)
        self.past_error = filtered_error
        return P + D + self.I
        # if __name__ == "__main__":
