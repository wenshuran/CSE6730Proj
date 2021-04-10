class drop:
    def __init__(self, x, y, scale=1.0, purity=1.0):
        scale = max(scale, 0.0)
        scale = min(scale, 1.0)
        self.x = x
        self.y = y
        self.amplitude = 255 * scale
        # TODO (low priority): 
        # purity: when drop is not pure water, surface's dampening factor
        # gets affected locally?
        self.purity = purity
        