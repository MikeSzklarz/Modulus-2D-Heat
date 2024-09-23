import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation # used if animation code is used POSSIBLY REMOVE
from sympy import Symbol

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.utils.io.plotter import ValidatorPlotter
from modulus.sym.key import Key
from modulus.sym.eq.pdes.diffusion import Diffusion

class CustomValidatorPlotter(ValidatorPlotter):
    def __init__(self, nx, ny, num_time_steps, size=50, num_plots=4):
        """
        Initialize the plotter with grid dimensions and number of time steps.
        """
        self.nx = nx
        self.ny = ny
        self.num_time_steps = num_time_steps
        self.size = size
        self.num_plots = num_plots
        
    def __call__(self, invar, true_outvar, pred_outvar):
        """
        Custom plotting function for validator that handles time-dependent 2D data.
        It interpolates the data onto a regular grid using _interpolate_2D from inherited ValidatorPlotter.
        """
        # extract x, y, and t coordinates
        x, y = invar["x"][:, 0], invar["y"][:, 0]
        t = invar["t"][:, 0]

        # get unique time steps and select ones to plot
        time_steps = np.unique(t)
        max_plots = min(self.num_plots, len(time_steps))
        time_step_indices = np.linspace(0, len(time_steps) - 1, max_plots, dtype=int)

        figures = []
        for i, time_step_index in enumerate(time_step_indices):
            t_val = time_steps[time_step_index]  # get the time value for this step

            # extract indices corresponding to the current time step
            # using tolerance to handle possible float errors
            tolerance = 1e-5
            indices = np.where(np.abs(t - t_val) < tolerance)[0]
            if len(indices) == 0:
                continue

            # prepare invar and outvars for interpolation
            # i is the index of the current time step
            invar_i = {
                'x': invar['x'][indices],
                'y': invar['y'][indices],
            }

            true_outvar_i = {k: v[indices] for k, v in true_outvar.items()}
            pred_outvar_i = {k: v[indices] for k, v in pred_outvar.items()}

            # interpolate using the inherited _interpolate_2D method from ValidatorPlotter
            # size is defined but creates problems with interpolation depending on 
            extent, true_interp, pred_interp = self._interpolate_2D(
                self.size, invar_i, true_outvar_i, pred_outvar_i
            )

            var_key = list(true_outvar.keys())[0] # 'u' in this case
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"2D Heat Transfer at t={t_val:.2f}")

            # TODO: POSSIBLY CHANGE VMIN AND VMAX VALUES (DIFFERENCE GIVES MORE INFO)
            im0 = axs[0].imshow(true_interp[var_key].T, origin="lower", extent=extent, vmin=0, vmax=1)
            axs[0].set_title(f"True {var_key} at t={t_val:.2f}")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("y")
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(pred_interp[var_key].T, origin="lower", extent=extent, vmin=0, vmax=1)
            axs[1].set_title(f"Predicted {var_key} at t={t_val:.2f}")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("y")
            fig.colorbar(im1, ax=axs[1])

            diff_grid = np.abs(true_interp[var_key] - pred_interp[var_key])
            im2 = axs[2].imshow(diff_grid.T, origin="lower", extent=extent)
            axs[2].set_title(f"Abs Difference (True - Pred) at t={t_val:.2f}")
            axs[2].set_xlabel("x")
            axs[2].set_ylabel("y")
            fig.colorbar(im2, ax=axs[2])

            plt.tight_layout()
            figures.append((fig, f"plot_time_{t_val:.2f}"))
                
        return figures

# true solution computation using finite difference
# TODO: MAKE MORE EXPLICIT HARD TO FOLLOW BECAUSE OF INDEXING
def update_temperature(T, alpha, dx, dy, dt):
    T_new = T.copy()

    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * (
        (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
        (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
    )

    return T_new

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # parameters for simulation
    Lx, Ly = 1.0, 1.0
    max_time = 1.0                  
    nx, ny = 50, 50                 # grid size
    dx = Lx / (nx - 1)              # grid spacing in x
    dy = Ly / (ny - 1)              # grid spacing in y
    alpha = 0.1                     # diffusion coefficient
    num_time_steps = 1000 
    dt = max_time / num_time_steps  # Time step size

    # initialize fixed boundary conditions
    T = np.zeros((nx, ny))
    T[0, :] = 1             # top boundary
    T[-1, :] = 1            # bottom boundary
    T[:, 0] = 1             # left boundary
    T[:, -1] = 1            # right boundary

    # computing true solution with update temp method
    # copying the initial state to the true solution
    u_true = [T.copy()]
    for _ in range(num_time_steps):
        T = update_temperature(T, alpha, dx, dy, dt)
        u_true.append(T.copy())

    u_true_array = np.array(u_true)  # shape: (num_time_steps + 1, nx, ny)

    x_vals = np.linspace(0, Lx, nx)
    y_vals = np.linspace(0, Ly, ny)
    t_vals = np.linspace(0, max_time, num_time_steps + 1)
    X, Y, T_mesh = np.meshgrid(x_vals, y_vals, t_vals, indexing='ij')

    X_flat = np.expand_dims(X.flatten(), axis=-1)  # shape: (nx * ny * (num_time_steps + 1), 1)
    Y_flat = np.expand_dims(Y.flatten(), axis=-1)  # shape: (nx * ny * (num_time_steps + 1), 1)
    T_flat = np.expand_dims(T_mesh.flatten(), axis=-1)  # shape: (nx * ny * (num_time_steps + 1), 1)

    # reshape u_true_array (num_time_steps + 1, nx, ny) to (nx * ny * (num_time_steps + 1), 1)
    # due to how the validator used pred outvar 
    u_true_flat = u_true_array.transpose(1, 2, 0).flatten()[:, None]

    assert X_flat.shape[0] == Y_flat.shape[0] == T_flat.shape[0] == u_true_flat.shape[0]

    # preparing invar and outvar for validator
    invar_numpy = {"x": X_flat, "y": Y_flat, "t": T_flat}
    outvar_numpy = {"u": u_true_flat}
        
###########################################################################

#TODO REMOVE CODE: POSSIBLY USE THIS TO SEE ANIMATION OF TIMESTEPS (HAVE TO CLOSE PLOT WINDOW TO CONTINUE)

    # fig, ax = plt.subplots()
    # cax = ax.imshow(u_true[0], cmap='turbo', origin='lower', vmin=0, vmax=1)
    # fig.colorbar(cax)

    # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    # def animate(frame):
    #     T_frame = u_true[frame]
    #     time = frame * dt

    #     cax.set_array(T_frame)
    #     time_text.set_text(f'Time: {time:.2f} s')
        
    #     return [cax, time_text]

    # ani = animation.FuncAnimation(fig, animate, frames=num_time_steps, interval=50, blit=True)

    # plt.show()

###########################################################################    

    
    diff = Diffusion(T="u", D=alpha, dim=2, time=True)
    diff_net = instantiate_arch(input_keys=[Key("x"), Key("y"), Key("t")],
                                output_keys=[Key("u")],
                                cfg=cfg.arch.fully_connected)
    
    nodes = diff.make_nodes() + [diff_net.make_node(name="diffusion_network")]
    
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    geo = Rectangle((0, 0), (Lx, Ly))

    domain = Domain()

    # explicitely defined time ranges
    # possible issues with skipping first time step with time range
    time_range_pde = {t_symbol: (0.0, max_time)}
    time_range_bc = {t_symbol: (0.0, max_time)}

    # initial condition interior contraint for t=0
    IC_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.IC_interior,
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC_interior, "IC_interior")

    # initial condition boundary contraint for t=0
    IC_boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1},
        batch_size=cfg.batch_size.IC_boundary,
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC_boundary, "IC_boundary")
    
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 1},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range_bc,
    )
    domain.add_constraint(BC, "BC")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"diffusion_u": 0},
        batch_size=cfg.batch_size.PDE_interior,
        parameterization=time_range_pde,
    )
    domain.add_constraint(interior, "interior")
        
    # plots to tensor board various metrics
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        plotter=CustomValidatorPlotter(nx=nx, ny=ny, num_time_steps=num_time_steps, num_plots=4),
    )
    domain.add_validator(validator)
    
    slv = Solver(cfg, domain)
    slv.solve()

if __name__ == "__main__":
    run()
