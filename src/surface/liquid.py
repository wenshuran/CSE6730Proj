import numpy as np
class liquid:
    def __init__(self, height, width, dampening=0.95):
        self.height = height
        self.width = width
        self.dampening = 0.95
        self.value = np.zeros((self.height, self.width), dtype=float)
        # TODO: store self.value in self.history everytime it gets update
        # scipy.sparse may be needed to save memory
        # self.history = np.array([])
        
        
    def clear(self):
        self.value = np.zeros(self.height, self.width)
        return self.value
    
    def clear_region(self, hstart, wstart, hend, wend):
        # TODO: clear rectangular region
        pass
        
    def shape(self):
        return (self.height, self.width)
    
    def inspect(self):
        return self.value
    
    def take_drops(self, drops):
        # TODO: if drops is only one drop object instead of array of drops
        # automatically handle the case
        """
        The surface gets some drops and update the canvas.

        Parameters
        ----------
        drops : arr
            input arr of drops. Each drop has its own location and amplitute

        Returns
        -------
        N/A

        See also
        -------
        /water/drop.py
        """
        for drop in drops:
            try: 
                self.value[drop.x][drop.y] = drop.amplitude
            except:
                raise Exception("can not set drop at {},{} with value {}".format(drop.x, drop.y, drop.amplitude))
                
    def update_one_step(self):
        # TODO: 4 edges with special care
        # TODO: vectorization to speed things up
        curr = np.copy(self.inspect())
        nxt = np.copy(self.inspect())
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                nxt[i][j] = (curr[i-1][j]+curr[i+1][j]+curr[i][j-1]+curr[i][j+1])/2 - curr[i][j]
                nxt[i][j] = nxt[i][j] * self.dampening
        self.value = nxt
        return self.inspect()
        
        
    def update_n_step(self, n=1):
        for i in range(n):
            self.update_one_step()
        return self.inspect()
    
    def record():
        # TODO: store self.value in self.history everytime it gets update
        # scipy.sparse may be needed to save memory
        pass
    
    def history(n=None):
        # TODO: retrieve records at step n
        # if n is None, retrieve all records
        pass
        # if n is not None:
        #     return self.history(n)
        # return self.history()
    
        
        
