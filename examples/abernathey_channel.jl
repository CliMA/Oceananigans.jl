# # Differentiating an Abernathey channel with Reactant and Enzyme
#
# This example builds a re-entrant, eddying channel in the spirit of
# [Abernathey, Marshall and Ferreira (2011)](https://doi.org/10.1175/2011JPO4708.1) --
# a workhorse idealization of the Antarctic Circumpolar Current -- and then
# *differentiates* a scalar diagnostic of the flow with respect to the model's
# initial condition.
#
# The simulation runs on Oceananigans' `ReactantState` architecture, which traces
# the time-stepping kernels into an [XLA](https://openxla.org) program. Because the
# whole simulation becomes a single differentiable program, we can hand it to
# [Enzyme](https://enzyme.mit.edu) and obtain the gradient of an objective function
# through the entire integration via reverse-mode automatic differentiation (AD).
#
# Physically, the channel captures the essential ingredients of the Southern Ocean:
#
#   * a meridional surface buoyancy gradient imposed by a buoyancy flux,
#   * an eastward wind stress that drives a northward Ekman transport and tilts isopycnals,
#   * a topographic ridge that allows a meridional pressure gradient to balance the zonal flow,
#   * and mesoscale eddies (resolved here) that flatten isopycnals and set the stratification.
#
# The differentiable diagnostic we use is the domain-averaged squared buoyancy,
# a proxy for the tracer variance, and we compute its sensitivity to the initial
# buoyancy field.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Reactant, Enzyme"
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: ynode, znode
using Oceananigans.Architectures: ReactantState

using Reactant
using Enzyme
using CUDA

# We work in double precision throughout, which AD through a baroclinic flow benefits from.

Oceananigans.defaults.FloatType = Float64

# ## Grid
#
# The domain is a zonally periodic channel, `Lx` wide and `Ly` long, that is
# `Bounded` in the meridional and vertical directions. We stretch the vertical grid
# so that resolution is finest near the surface, where the buoyancy forcing and wind
# stress act.

Lx = 1000kilometers # zonal (east-west) domain length [m]
Ly = 2000kilometers # meridional (north-south) domain length [m]

architecture = ReactantState()

Nx, Ny, Nz = 96, 192, 32

## Geometrically stretched vertical spacing: thin cells at the surface, thick at depth
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)
Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz + 1] = 0

underlying_grid = RectilinearGrid(architecture,
                                  topology = (Periodic, Bounded, Bounded),
                                  size = (Nx, Ny, Nz),
                                  halo = (4, 4, 4),
                                  x = (0, Lx),
                                  y = (0, Ly),
                                  z = z_faces)

# A meridional ridge breaks the zonal symmetry. The ridge is a Gaussian bump in `x`
# that rises nearly to the surface, punctured by a deep gap in the southern half of the
# channel so the flow can still circulate -- analogous to Drake Passage.

function ridge(x, y)
    zonal = (Lz + 100) * exp(-(x - Lx/2)^2 / 1e6kilometers)
    gap   = 1 - 0.5 * (tanh((y - Ly/6) / 1e5) - tanh((y - Ly/2) / 1e5))
    return zonal * gap - Lz
end

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))

# ## Physical parameters and boundary conditions
#
# We collect the forcing parameters into a named tuple that gets threaded through the
# boundary-condition and forcing functions below.

α  = 2e-4    # [K⁻¹]    thermal expansion coefficient
g  = 9.8061  # [m s⁻²]  gravitational acceleration
cᵖ = 3994.0  # [J K⁻¹]  heat capacity
ρ  = 999.8   # [kg m⁻³] reference density

parameters = (; Ly, Lz,
              Qᵇ = 10 / (ρ * cᵖ) * α * g,  # buoyancy flux magnitude [m² s⁻³]
              y_shutoff = 5/6 * Ly,        # latitude north of which the buoyancy flux vanishes [m]
              τ = 0.2 / ρ,                 # peak kinematic wind stress [m² s⁻²]
              μ = 1 / 30days,              # bottom-drag damping rate [s⁻¹]
              ΔB = 8 * α * g,              # surface buoyancy contrast [m s⁻²]
              h = 1000.0,                  # stratification decay scale [m]
              y_sponge = 19/20 * Ly,       # southern edge of the northern sponge [m]
              λt = 7days)                  # sponge relaxation timescale [s]

# A cosine-shaped surface buoyancy flux warms the south and cools the north, switching
# off near the northern wall.

@inline function buoyancy_flux(i, j, grid, clock, fields, p)
    y = ynode(j, grid, Center())
    return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
end

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(buoyancy_flux, discrete_form=true, parameters=parameters))

# An eastward wind stress is applied at the surface as a `Field` boundary condition;
# we set its profile later. Linear bottom drag removes the momentum the wind injects.

u_stress_bc = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

@inline u_drag(i, j, grid, clock, fields, p) = @inbounds -p.μ * p.Lz * fields.u[i, j, 1]
@inline v_drag(i, j, grid, clock, fields, p) = @inbounds -p.μ * p.Lz * fields.v[i, j, 1]

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = FluxBoundaryCondition(u_drag, discrete_form=true, parameters=parameters))
v_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(v_drag, discrete_form=true, parameters=parameters))

# ## Coriolis
#
# A southern-hemisphere β-plane gives the planetary vorticity gradient that supports
# the Rossby waves and the asymmetry between the eastward and westward jets.

coriolis = BetaPlane(f₀ = -1e-4, β = 1e-11)

# ## Forcing
#
# Near the northern wall we relax buoyancy back to an exponential target profile.
# This "sponge" mimics the deep stratification imported from the rest of the ocean and
# prevents the channel from drifting away from a realistic state.

@inline target_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline sponge_mask(y, p) = max(0.0, y - p.y_sponge) / (p.Ly - p.y_sponge)

@inline function buoyancy_relaxation(i, j, k, grid, clock, fields, p)
    y = ynode(j, grid, Center())
    z = znode(k, grid, Center())
    b = @inbounds fields.b[i, j, k]
    return -1 / p.λt * sponge_mask(y, p) * (b - target_buoyancy(z, p))
end

Fb = Forcing(buoyancy_relaxation, discrete_form=true, parameters=parameters)

# ## Closure
#
# Small horizontal and vertical diffusivities/viscosities provide the dissipation that
# the resolved eddy field needs at the grid scale.

horizontal_closure = HorizontalScalarDiffusivity(ν=30.0, κ=0.5e-5)
vertical_closure   = VerticalScalarDiffusivity(ν=3e-4, κ=0.5e-5)

# ## Model
#
# We use a `HydrostaticFreeSurfaceModel` with a split-explicit free surface, WENO
# advection for momentum and the single buoyancy tracer `b`, and `BuoyancyTracer`
# buoyancy (so `b` *is* the buoyancy).

model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                    free_surface = SplitExplicitFreeSurface(substeps=500),
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    closure = (horizontal_closure, vertical_closure),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs, v=v_bcs),
                                    forcing = (; b=Fb))

# ## Initial conditions
#
# The wind stress is a half-sine in latitude, eastward everywhere. The initial
# buoyancy is the same exponential profile we relax toward, plus a touch of noise to
# seed instability.

wind_stress = Field{Face, Center, Nothing}(grid)
set!(wind_stress, (x, y) -> -parameters.τ * sin(π * y / parameters.Ly))

bᵢ = Field{Center, Center, Center}(grid)
set!(bᵢ, (x, y, z) -> target_buoyancy(z, parameters) + 1e-8 * randn())

# ## The forward problem
#
# `run_channel!` installs the initial condition and wind stress, resets the clock, and
# steps the model forward. Reactant requires the time-stepping loop to be traceable, so
# we wrap the fixed number of steps in `@trace` rather than using a `Simulation` with
# callbacks.

Δt = 5minutes
model.clock.last_Δt = Δt

function step_channel!(model)
    Δt = model.clock.last_Δt
    @trace mincut=true track_numbers=false for _ in 1:10
        time_step!(model, Δt)
    end
    return nothing
end

function run_channel!(model, bᵢ, wind_stress)
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)
    set!(model.tracers.b, bᵢ)

    model.clock.iteration = 0
    model.clock.time = 0

    step_channel!(model)

    return nothing
end

# Our objective is the domain-averaged squared buoyancy after the integration -- a
# single number that summarizes the tracer field. Differentiating it tells us how each
# point of the *initial* buoyancy influences the *final* buoyancy variance, sensitivity
# information that flows backward through every time step.

function estimate_tracer_error(model, bᵢ, wind_stress)
    run_channel!(model, bᵢ, wind_stress)
    Nx, Ny, Nz = size(model.grid)
    b² = parent(model.tracers.b) .^ 2
    return sum(b²) / (Nx * Ny * Nz)
end

# ## The differentiable problem
#
# Reverse-mode AD needs a "shadow" for every differentiated argument to accumulate the
# adjoint (the derivative). We seed them to zero and wrap each primal/shadow pair in
# `Duplicated`; the scalar objective is `Active`. `set_strong_zero` ensures the shadows
# start cleanly at zero before the reverse pass accumulates into them.

function differentiate_tracer_error(model, bᵢ, wind_stress, dmodel, dbᵢ, dwind_stress)
    autodiff(set_strong_zero(Enzyme.Reverse),
             estimate_tracer_error, Active,
             Duplicated(model, dmodel),
             Duplicated(bᵢ, dbᵢ),
             Duplicated(wind_stress, dwind_stress))

    return dbᵢ
end

dmodel = Enzyme.make_zero(model)
dbᵢ = Field{Center, Center, Center}(grid)
dwind_stress = Field{Face, Center, Nothing}(grid)

# ## Compile the gradient
#
# `@compile` traces `differentiate_tracer_error` into an optimized XLA executable.
# `raise=true` lowers Oceananigans' kernels to Reactant's intermediate representation, and
# `sync=true` makes the call blocking.

@info "Compiling the gradient program (this can take a few minutes)..."

rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true differentiate_tracer_error(model, bᵢ, wind_stress,
                                                                                                       dmodel, dbᵢ, dwind_stress)

# Running it integrates the channel forward and pulls the adjoint back through every time
# step, filling `dbᵢ` with ``\partial J / \partial b_i`` at every grid point.

rdifferentiate_tracer_error(model, bᵢ, wind_stress, dmodel, dbᵢ, dwind_stress)

@info "Computed dJ/dbᵢ; ‖dJ/dbᵢ‖∞ = $(maximum(abs, parent(dbᵢ)))"

# `dbᵢ` is now an ordinary Oceananigans `Field` holding the sensitivity map: regions where
# it is large are the parts of the initial buoyancy field that most strongly control the
# final tracer variance -- the building block for gradient-based optimization and data
# assimilation.
