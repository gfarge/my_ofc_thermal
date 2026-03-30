"""
Implementation of the Olami, Feder, Christensen (OFC) model, 
with an added term of thermal activation.
"""

# Imports
import numpy as np
import pandas as pd
import pickle

class ThermalOFC():
    def __init__(self, Nx=500, d=4e-5, delta_th=0.1, v=1.0, theta=2.76e-5, omega_0=1e13, random_seed=None):
        # Set random seed
        np.random.seed(random_seed)

        # Store parameters
        self.Nx = Nx  # grid size in one dimension (Nx x Nx)
        self.delta_th = delta_th  # std of stress threshold distribution [stress]
        self.d = d  # dissipation [.]
        self.v = v  # loading velocity [length/time]
        self.theta = theta  # thermal activation scale [stress]
        self.omega_0 = omega_0  # attempt frequency [1/time]

        # Derived parameters
        self.K =  4*self.d / (1 - self.d)  # driving medium stiffness [stress/length]
        self.loading_rate = self.v * self.K  # tectonic loading rate [stress/time]

        # Initialize stress grid
        self.thresholds = 1.0 + np.random.randn(Nx, Nx) * self.delta_th
        self.stress = np.random.rand(Nx, Nx) * self.thresholds  # initial stress values [stress] (below thresholds)
        self.neighbors = self.compute_neighbors()

        # Initialize time and logging
        self.time = 0.0  # current time [time]
        self.log_time = -np.inf  # log of current time, initialized to -inf for log-space creep
        self.logged_states = []  # list to store logged states
        self.catalog = Catalog()  # event catalog
    
    def summary(self):
        print("Thermal OFC model summary:")
        print(f"  Grid size: {self.Nx} x {self.Nx}\n")
        print(f"  Dissipation (d): {self.d:.1e}")
        print(f"  Threshold disorder (delta_th): {self.delta_th:.1e}")
        print(f"  Loading velocity (v): {self.v:.1e} length/time")
        print(f"  Derived stiffness (K): {self.K:.2e} stress/length")
        print(f"  Derived loading rate: {self.loading_rate:.2e} stress/time\n")
        print(f"  Attempt frequency (omega_0): {self.omega_0:.1e} 1/time")
        print(f"  Thermal activation scale (theta): {self.theta:.1e} stress\n")
        print(f"  Time scale ratio (omega_0 * theta / loading_rate): {self.omega_0 * self.theta / self.loading_rate:.2e} (dimensionless)")

    def step_athermal(self):
        """Advance the simulation by one time step : loading + event."""
        # Compute the "excitation spectrum": the stress gap between local threshold and stress
        self.stress_gaps = self.thresholds - self.stress  # stress gap [stress]
        if np.any(self.stress_gaps < 0):
            print("Some negative stress gaps detected")
        # self.stress_gaps = np.maximum(self.stress_gaps, 0.0)  # ensures non-negative gaps (numerical errors)

        # Time to ATHERMAL event : find the minimum stress gap, compute delta_t
        min_gap = np.min(self.stress_gaps)  # minimum stress gap [stress]
        delta_t_athermal = min_gap / self.loading_rate  # time to next event [time]

        rupture_type = 'athermal'
        delta_t = delta_t_athermal  # select time
        ii_init, jj_init = np.unravel_index(np.argmin(self.stress_gaps), self.stress.shape)  # i, j indices of initial failing cell

        # Advance time and apply tectonic loading
        self.time += delta_t
        self.log_time = np.log(self.time)  # update log_time for consistency, even though it's not used in athermal steps
        self.stress += self.loading_rate * delta_t

        # Run avalanche cascade
        event = self._run_avalanche(ii_init, jj_init, rupture_type)

        # Event logging
        self.catalog.add_event(event)
        
        return event, self.time
    
    def creep(self):
        """Let the system creep for one time step: the system evolves only through thermal activation, without tectonic loading."""
        # Compute the "excitation spectrum": the stress gap between local threshold and stress
        self.stress_gaps = self.thresholds - self.stress  # stress gap [stress]
        if np.any(self.stress_gaps < 0):
            print("Some negative stress gaps detected")
        # self.stress_gaps = np.maximum(self.stress_gaps, 0.0)  # ensures non-negative gaps (numerical errors)

        # Time to THERMAL event : activation spectrum and delta_t sampling
        activation_values = self.omega_0 * np.exp(-self.stress_gaps / self.theta)  # activation spectrum (rate of thermal events for each cell)
        if np.all(activation_values == 0):  # checking for infinite waiting times
            print("Terminating creep: no thermal events possible")
            return None, np.inf

        # Compute time to thermal event for each cell and select the minimum
        u = np.random.rand(self.Nx, self.Nx)  # uniform variable for sampling
        delta_t_thermal = - 1 / activation_values * np.log(u)  # time to thermal event for each cell [time], computed for Poisson process with rate given by activation spectrum

        rupture_type = 'thermal'
        delta_t = np.min(delta_t_thermal)  # select time
        initiating_idx = np.argmin(delta_t_thermal)  # index of cell that initiates the thermal event (for flat array)
        ii_init, jj_init = np.unravel_index(initiating_idx, (self.Nx, self.Nx))  # i, j indices of initial failing cell

        # Advance time (no tectonic loading)
        self.time += delta_t
        self.log_time = np.log(self.time)  # update log_time for consistency

        # Run avalanche cascade
        event = self._run_avalanche(ii_init, jj_init, rupture_type)

        # Event logging
        self.catalog.add_event(event)
        
        return event, self.time
    
    def creep_logtime(self):
        """
        Let the system creep for one time step: the system evolves only through
        thermal activation, without tectonic loading.

        All time/rate computations are performed in log-space to handle the extreme
        dynamic range of activation rates when stress gaps are large relative to θ.

        δt = -1/λ * ln(u)  =>  log(δt) = -log(λ) + log(-log(u))
        where log(λ) = log(ω_0) - stress_gap / θ

        Time is tracked as self.log_time = log(current_time) to avoid overflow
        when individual δt values are astronomically large.
        """
        # --- Stress gap: x_i = threshold_i - stress_i ---
        self.stress_gaps = self.thresholds - self.stress  # shape (Nx, Nx) [stress units]
        if np.any(self.stress_gaps < 0):
            print("Some negative stress gaps detected")

        # --- Log-space activation rates: log(λ_i) = log(ω_0) - gap_i / θ ---
        log_activation = np.log(self.omega_0) - self.stress_gaps / self.theta  # shape (Nx, Nx)

        # Guard: if ALL rates are effectively -inf (no event possible in float64)
        if np.all(log_activation < -745):
            print("Terminating creep: no thermal events possible (all rates underflow)")
            return None, self.log_time

        # --- Log time-to-event: log(δt_i) = -log(λ_i) + log(-log(u_i)) ---
        u = np.clip(np.random.rand(self.Nx, self.Nx), 1e-300, 1)  # U(0,1), guarded against exact 0
        log_exp_sample = np.log(-np.log(u))             # log of Exp(1) sample, shape (Nx, Nx)
        log_delta_t    = -log_activation + log_exp_sample  # log(δt) per cell, shape (Nx, Nx)

        # --- Select minimum δt, equivalently minimum log(δt) ---
        initiating_idx   = np.argmin(log_delta_t)
        ii_init, jj_init = np.unravel_index(initiating_idx, (self.Nx, self.Nx))
        log_dt_min       = log_delta_t[ii_init, jj_init]

        # --- Advance log_time via log-sum-exp: log(t + δt) = log(t) + log(1 + exp(log(δt) - log(t))) ---
        # Special cases: if either term dominates by many orders of magnitude, log1p(exp(...)) 
        # reduces to the dominant term, which is numerically exact.
        if np.isneginf(self.log_time):
            # Edge case: t = 0 at simulation start, log(0) = -inf; time is just δt
            self.log_time = log_dt_min
            self.time = +np.inf  # time in linear space is not tracked during creep_logtime to avoid overflow
        else:
            # General case: log(t + δt) via log-sum-exp
            log_sum_max   = max(self.log_time, log_dt_min)
            log_sum_min   = min(self.log_time, log_dt_min)
            self.log_time = log_sum_max + np.log1p(np.exp(log_sum_min - log_sum_max))
            self.time = +np.inf  # time in linear space is not tracked during creep_logtime to avoid overflow

        rupture_type = 'thermal'

        # --- Run avalanche cascade from initiating cell ---
        event = self._run_avalanche(ii_init, jj_init, rupture_type)

        # --- Event logging ---
        self.catalog.add_event(event)

        return event, self.log_time

    def step(self):
        """Advance the simulation by one time step : loading + event."""
        # Compute the "excitation spectrum": the stress gap between local threshold and stress
        self.stress_gaps = self.thresholds - self.stress  # stress gap [stress]
        if np.any(self.stress_gaps < 0):
            print("Some negative stress gaps detected")
        #self.stress_gaps = np.maximum(self.stress_gaps, 0.0)  # ensures non-negative gaps (numerical errors)

        # Time to ATHERMAL event : find the minimum stress gap, compute delta_t
        min_gap = np.min(self.stress_gaps)  # minimum stress gap [stress]
        delta_t_athermal = min_gap / self.loading_rate  # time to next event [time]

        # Time to THERMAL event : activation spectrum and delta_t sampling
        exp_values = np.exp(-self.stress_gaps / self.theta)  # activation spectrum
        exp_sum = np.sum(exp_values)  # sum over activation spectrum
        u = np.random.rand()  # uniform variable for sampling
        thermal_argument = self.loading_rate / (self.omega_0 * self.theta) * (1 / exp_sum) * np.log(u)  # always negative
        delta_t_thermal = self.theta / self.loading_rate * np.log(1 - thermal_argument)  # time to thermal event [time]

        # Run thermal or athermal event based on which occurs first
        if delta_t_athermal < delta_t_thermal:
            rupture_type = 'athermal'
            delta_t = delta_t_athermal  # select time
            ii_init, jj_init = np.unravel_index(np.argmin(self.stress_gaps), self.stress.shape)  # i, j indices of initial failing cell

        else:
            rupture_type = 'thermal'
            delta_t = delta_t_thermal  # select time
            probs = (exp_values / exp_sum).ravel()
            initiating_idx = np.random.choice(self.Nx * self.Nx, p=probs)
            ii_init, jj_init = np.unravel_index(initiating_idx, (self.Nx, self.Nx))

        # Advance time and apply tectonic loading
        self.time += delta_t
        self.log_time = np.log(self.time)  # update log_time for consistency
        self.stress += self.loading_rate * delta_t

        # Run avalanche cascade
        event = self._run_avalanche(ii_init, jj_init, rupture_type)

        # Event logging
        self.catalog.add_event(event)
        
        return event, self.time

    def _run_avalanche(self, ii_init, jj_init, rupture_type):
        """Run an avalanche starting from the initial failing cell."""
        
        transfer_fraction = (1 - self.d) / 4
        
        # Event tracking
        failed_ii = []
        failed_jj = []
        stress_drops = []
                
        # Avalanche
        # --> Break initial cell
        failed_ii.append(ii_init)  # record failure
        failed_jj.append(jj_init)
        
        stress_at_failure = self.stress[ii_init, jj_init].copy()  # record stress drop based on actual stress
        stress_drops.append(stress_at_failure)  # record stress drop

        neighbors_fail = self.neighbors[ii_init, jj_init]  # get neighbors
        ii_nb = neighbors_fail[:, 0]
        jj_nb = neighbors_fail[:, 1]

        transfer = transfer_fraction * stress_at_failure  # compute transfer based on actual stress

        self.stress[ii_init, jj_init] = 0.0  # reset stress of failed cell
        np.add.at(self.stress, (ii_nb, jj_nb), transfer)  # transfer stress to neighbors
        self.thresholds[ii_init, jj_init] = 1.0 + np.random.randn() * self.delta_th  # reset threshold of failed cell (for annealed disorder)

        #  --> Loop until no more failures
        while True:
            above_threshold = self.stress >= self.thresholds
            
            if not np.any(above_threshold):
                break
            
            ii_fail, jj_fail = np.where(above_threshold)
            
            # Record failures
            failed_ii.extend(ii_fail)
            failed_jj.extend(jj_fail)
            
            # Transfer actual stress, not threshold
            stress_at_failure = self.stress[ii_fail, jj_fail].copy()
            stress_drops.extend(stress_at_failure)
            
            # Get neighbors
            neighbors_fail = self.neighbors[ii_fail, jj_fail]
            ii_nb = neighbors_fail[:, :, 0].ravel()
            jj_nb = neighbors_fail[:, :, 1].ravel()
            
            # Transfer based on actual stress
            transfers = np.repeat(transfer_fraction * stress_at_failure, 4)
            
            # Update stresses
            self.stress[ii_fail, jj_fail] = 0.0
            np.add.at(self.stress, (ii_nb, jj_nb), transfers)
            
            # Reset threshold of failed cells (for annealed disorder)
            self.thresholds[ii_fail, jj_fail] = 1.0 + np.random.randn(len(ii_fail)) * self.delta_th
                
        event = {
            "time": self.time,
            "log_time": self.log_time,
            "rupture_type": rupture_type,
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
    
    def log_state(self):
        """Log the current state of the system (for visualization or analysis)."""
        state = {
            "time": self.time,
            "stress": self.stress.copy(),
            "thresholds": self.thresholds.copy()
        }
        self.logged_states.append(state)

    def save(self, dir=None, filename=None):
        """Save the model"""
        # Deal with path
        if dir is None:
            dir = "./"
        if filename is None: # default filename based on parameters
            filename = f"tofc_N{self.Nx:d}D{self.d:.1e}V{self.v:.1e}T{self.theta:.1e}O{self.omega_0:.1e}.pkl"
        path = dir + filename

        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
class Catalog():
    """Catalog of events produced by the model."""
    def __init__(self):
        self.events = []
    
    def add_event(self, event):
        self.events.append(event)

    def _process(self):
        for event in self.events:
            event["size"] = len(event["ii"])
            event["mag"] = 2/3 * np.log10(event["stress_drops"].sum())
            if np.isneginf(event["log_time"]):
                event["log_time"] = np.log(event["time"])

    def as_df(self):
        """Return the catalog as a pandas DataFrame."""
        self._process()  # process events to add size and magnitude
        self.df = pd.DataFrame(self.events)

    def summary(self):
        """Print a few stats about the catalog"""
        self.as_df()
        print("\nCatalog stats:")
        print(f"  Time span: {self.df['time'].min():.2f} - {self.df['time'].max():.2f}")
        print(f"  Number of events: {len(self.df)}")
        print(f"  Min-Max event size: {self.df['size'].min()} - {self.df['size'].max()}")
        print(f"  Min-Max magnitude: {self.df['mag'].min():.2f} - {self.df['mag'].max():.2f}")
        print("  Proportion of event types: ")
        print(self.df['rupture_type'].value_counts(normalize=True))