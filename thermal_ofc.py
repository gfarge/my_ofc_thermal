"""
Implementation of the Olami, Feder, Christensen (OFC) model, 
with an added term of thermal activation.
"""

# Imports
import numpy as np

class ThermalOFC():
    def __init__(self, Nx=500, d=4e-5, delta_th=0.1, v=1.0, random_seed=None):
        # Set random seed
        np.random.seed(random_seed)

        # Store parameters
        self.Nx = Nx  # grid size in one dimension (Nx x Nx)
        self.delta_th = delta_th  # std of stress threshold distribution [stress]
        self.d = d  # dissipation [.]
        self.v = v  # loading velocity [length/time]

        # Derived parameters
        self.K =  4*self.d / (1 - self.d)  # driving medium stiffness [stress/length]
        self.loading_rate = self.v * self.K  # tectonic loading rate [stress/time]

        # Initialize stress grid
        self.thresholds = 1.0 + np.random.randn(Nx, Nx) * self.delta_th
        self.stress = np.random.rand(Nx, Nx) * self.thresholds  # initial stress values [stress] (below thresholds)
        self.neighbors = self.compute_neighbors()

        # Initialize time and logging
        self.time = 0.0  # current time [time]
        self.logged_states = []  # list to store logged states

    def step(self):
        """Advance the simulation by one time step : loading + event."""
        # Compute the "excitation spectrum": the stress gap between local threshold and stress
        self.stress_gaps = self.thresholds - self.stress  # stress gap [stress]
        self.stress_gaps = np.maximum(self.stress_gaps, 0.0)  # ensures non-negative gaps (numerical errors)

        # Find the minimum stress gap, compute time to next __ATHERMAL__ event
        min_gap = np.min(self.stress_gaps)  # minimum stress gap [stress]
        delta_t = min_gap / self.loading_rate  # time to next event [time]
        ii_init, jj_init = np.unravel_index(np.argmin(self.stress_gaps), self.stress.shape)  # indices of initial failing cell

        # Advance time and apply tectonic loading
        self.time += delta_t
        self.stress += self.loading_rate * delta_t

        # Run avalanche cascade
        event = self._run_avalanche(ii_init, jj_init)

        return event, self.time
    
    # def _run_avalanche(self, ii_init, jj_init):
    #     """Run an avalanche starting from the initial failing cell.

    #     Args:
    #         ii_init (int): Row index of failure nucleation.
    #         jj_init (int): Column index of nucleating cell.

    #     Returns:
    #         event (dict): Dictionary containing event information (time, initial cell, failed cells, stress drops).
    #     """
    
    #     # Pre-compute constant
    #     transfer_fraction = (1 - self.d) / 4

    #     # Event tracking variables
    #     failed_ii = []
    #     failed_jj = []
    #     stress_drops = []
    
    #     # Mark nucleation cell as needing to fail
    #     self.stress[ii_init, jj_init] = self.thresholds[ii_init, jj_init]
    
    #     # Avalanche loop
    #     while True:
    #         # Find all failing cells
    #         above_threshold = self.stress >= self.thresholds
        
    #         if not np.any(above_threshold):  # if none, exit the loop
    #             break
            
    #         ii_fail, jj_fail = np.where(above_threshold)  # indices of failing cells
        
    #         # Record failures
    #         failed_ii.extend(ii_fail)
    #         failed_jj.extend(jj_fail)
    #         stress_drops.extend(self.thresholds[ii_fail, jj_fail])
    
    #         # Get neighbor indices (n_fail, 4, 2) -> flatten to (n_fail * 4,)
    #         neighbors_fail = self.neighbors[ii_fail, jj_fail]  # (n_fail, 4, 2)
    #         ii_nb = neighbors_fail[:, :, 0].ravel()
    #         jj_nb = neighbors_fail[:, :, 1].ravel()
        
    #         # Compute transfers : (n_fail * 4,) same shape and order as ii_nb, jj_nb
    #         transfers = np.repeat(transfer_fraction * self.thresholds[ii_fail, jj_fail], 4)
        
    #         # Update stresses
    #         np.add.at(self.stress, (ii_nb, jj_nb), transfers)  # transfer to neighbors
    #         self.stress[ii_fail, jj_fail] = 0.0  # stress drop

    #         # Reset thresholds of failed cells (annealed disorder)
    #         self.thresholds[ii_fail, jj_fail] = 1.0 + np.random.randn(len(ii_fail)) * self.delta_th

    #     event = {"time": self.time,  # event time
    #              "ii0": ii_init,  # initial failing cell
    #              "jj0": jj_init,  # initial failing cell
    #              "ii": np.array(failed_ii),  # all failed cells
    #              "jj": np.array(failed_jj),  # all failed cells
    #              "stress_drops": np.array(stress_drops) # stress drops of failed cells
    #              }

    #     return event

    def _run_avalanche(self, ii_init, jj_init):
        """Run an avalanche starting from the initial failing cell."""
        
        transfer_fraction = (1 - self.d) / 4
        
        # Event tracking
        failed_ii = []
        failed_jj = []
        stress_drops = []
        
        # Initialize nucleation
        self.stress[ii_init, jj_init] = self.thresholds[ii_init, jj_init]
        
        # Track which cells have failed (for annealed disorder reset)
        ever_failed = np.zeros((self.Nx, self.Nx), dtype=bool)
        
        # Avalanche loop
        while True:
            above_threshold = self.stress >= self.thresholds
            
            if not np.any(above_threshold):
                break
            
            ii_fail, jj_fail = np.where(above_threshold)
            
            # Record failures
            failed_ii.extend(ii_fail)
            failed_jj.extend(jj_fail)
            
            # **FIX 1: Transfer actual stress, not threshold**
            stress_at_failure = self.stress[ii_fail, jj_fail].copy()
            stress_drops.extend(stress_at_failure)
            
            # Get neighbors
            neighbors_fail = self.neighbors[ii_fail, jj_fail]
            ii_nb = neighbors_fail[:, :, 0].ravel()
            jj_nb = neighbors_fail[:, :, 1].ravel()
            
            # **FIX 2: Transfer based on actual stress**
            transfers = np.repeat(transfer_fraction * stress_at_failure, 4)
            
            # Update stresses
            np.add.at(self.stress, (ii_nb, jj_nb), transfers)
            self.stress[ii_fail, jj_fail] = 0.0
            
            # **FIX 3: Track cells for post-avalanche threshold reset**
            ever_failed[ii_fail, jj_fail] = True
        
        # **FIX 4: Reset thresholds after avalanche completes**
        ii_reset, jj_reset = np.where(ever_failed)
        self.thresholds[ii_reset, jj_reset] = 1.0 + np.random.randn(len(ii_reset)) * self.delta_th
        
        event = {
            "time": self.time,
            "ii0": ii_init,
            "jj0": jj_init,
            "ii": np.array(failed_ii),
            "jj": np.array(failed_jj),
            "stress_drops": np.array(stress_drops)
        }
        
        return event

    def compute_neighbors(self):
        """Compute the neighbor indices for each cell in the grid for efficiency."""
        neighbors = np.zeros((self.Nx, self.Nx, 4, 2), dtype=int)
        for ii in range(self.Nx):
            for jj in range(self.Nx):
                # Top, bottom, left, right with periodic BC
                neighbors[ii, jj, 0] = [(ii - 1) % self.Nx, jj]      # top
                neighbors[ii, jj, 1] = [(ii + 1) % self.Nx, jj]      # bottom
                neighbors[ii, jj, 2] = [ii, (jj - 1) % self.Nx]      # left
                neighbors[ii, jj, 3] = [ii, (jj + 1) % self.Nx]      # right
        return neighbors
