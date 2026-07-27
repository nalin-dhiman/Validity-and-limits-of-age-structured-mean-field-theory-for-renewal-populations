class PDESafetyError(Exception):
    """
    Raised when PDE Diagnostics fail safety checks (Mass Conservation or Truncation).
    """
    def __init__(self, guard_name, severity, value, threshold, hint=""):
        self.guard_name = guard_name
        self.severity = severity # 'soft' or 'hard'
        self.value = value
        self.threshold = threshold
        self.hint = hint

        msg = f"PDE Safety Guard [{guard_name}] ({severity.upper()}) Failed: Value={value:.2e} > Threshold={threshold:.2e}. {hint}"
        super().__init__(msg)
