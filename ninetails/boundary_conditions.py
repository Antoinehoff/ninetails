import numpy as np

class BoundaryConditions:
    '''
    Apply boundary conditions. We do not have to handle x and y boundaries.
    In z, we can have periodic boundary conditions or twist and shift.

    The ghosts are placed at the end of the array, so the two lower ghosts are
        f[-2], f[-1]
    The two upper ghosts are
        f[n], f[n+1]
    where n is the number of grid points in z.

    '''
    def __init__(self, y, model='periodic', nghosts = 4):
        self.model = model
        # Get dimensions
        self.shape = np.shape(y[0])
        # Check if we have z dimension
        self.has_z = self.shape[-1] > 1
        self.nghosts = nghosts
        self.nz = self.shape[-1]

        if not self.has_z:
            self.apply = self.none_bc
        else:
            if self.model == 'periodic':
                self.apply = self.periodic
            if self.model == 'twist_and_shift':
                raise NotImplementedError('Boundary conditions not implemented yet.')
            else:
                raise NotImplementedError('Boundary conditions not recognized.')
        
    def periodic(self, y):
        for moment in y:
            # Apply periodic boundary conditions
            for i in range(1, self.nghosts/2 + 1):
                # lower ghosts
                moment[:, :, -i] = moment[:, :, self.nz-i]
                # upper ghosts
                moment[:, :, self.nz + (i-1)] = moment[:, :, i-1]

    def twist_and_shift(self, y):
        pass

    def none_bc(self, y):
        pass