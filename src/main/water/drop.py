class drop:
    def __init__(self, x, y, width=20, amplitude=1.0):
        self.x = x
        self.y = y
        self.width = width
        self.amplitude = amplitude
        
        # TODO (low priority): 
        # purity: when drop is not pure water, surface's dampening factor
        # gets affected locally?
        # self.__purity = purity
        