using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using SeawaterPolynomials.TEOS10
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper
using OrthogonalSphericalShellGrids

@inline ϕ²(i, j, k, grid, ϕ)    = @inbounds ϕ[i, j, k]^2
@inline spᶠᶜᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², Φ.v))
@inline spᶜᶠᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.v[i, j, k]^2 + ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², Φ.u))

@inline u_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.u[i, j, 1] * spᶠᶜᶜ(i, j, 1, grid, Φ)
@inline v_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.v[i, j, 1] * spᶜᶠᶜ(i, j, 1, grid, Φ)

# Keep a constant linear drag parameter independent on vertical level
@inline u_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, k] * spᶠᶜᶜ(i, j, k, grid, fields) 
@inline v_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, k] * spᶜᶠᶜ(i, j, k, grid, fields) 


grid = TripolarGrid(size = (100, 100, 50), z = (-1000, 0))
bottom_height(x, y) = - 500 * rand() - 500

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
closure = CATKEVerticalDiffusivity()
tracers = (:T, :S, :e)
free_surface = nothing
forcing = NamedTuple()
timestepper = :QuasiAdamsBashforth2
equation_of_state = TEOS10EquationOfState(; reference_density=1020)
coriolis = HydrostaticSphericalCoriolis()

# Detect whether we are on a single column grid
bottom_drag_coefficient = 1e-3
    
u_immersed_drag = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
v_immersed_drag = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

u_immersed_bc = ImmersedBoundaryCondition(bottom = u_immersed_drag)
v_immersed_bc = ImmersedBoundaryCondition(bottom = v_immersed_drag)

# Set up boundary conditions using Field
top_zonal_momentum_flux      = τx = Field{Face, Center, Nothing}(grid)
top_meridional_momentum_flux = τy = Field{Center, Face, Nothing}(grid)
top_ocean_heat_flux          = Jᵀ = Field{Center, Center, Nothing}(grid)
top_salt_flux                = Jˢ = Field{Center, Center, Nothing}(grid)

# Construct ocean boundary conditions including surface forcing and bottom drag
u_top_bc = FluxBoundaryCondition(τx)
v_top_bc = FluxBoundaryCondition(τy)
T_top_bc = FluxBoundaryCondition(Jᵀ)
S_top_bc = FluxBoundaryCondition(Jˢ)
    
u_bot_bc = FluxBoundaryCondition(u_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
v_bot_bc = FluxBoundaryCondition(v_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

default_boundary_conditions = (u = FieldBoundaryConditions(top=u_top_bc, bottom=u_bot_bc, immersed=u_immersed_bc),
                               v = FieldBoundaryConditions(top=v_top_bc, bottom=v_bot_bc, immersed=v_immersed_bc),
                               T = FieldBoundaryConditions(top=T_top_bc),
                               S = FieldBoundaryConditions(top=S_top_bc))

# Merge boundary conditions with preference to user
# TODO: support users specifying only _part_ of the bcs for u, v, T, S (ie adding the top and immersed
# conditions even when a user-bc is supplied).
buoyancy = SeawaterBuoyancy(; equation_of_state)
tracer_advection = (T=WENO(), S=WENO(), e=nothing)
# timestepper = QuasiAdamsBashforth2TimeStepper(grid, ([], []))

@info "constructing model"

@time begin
    ocean_model = HydrostaticFreeSurfaceModel(; grid,
                                                buoyancy,
                                                closure,
                                                tracer_advection,
                                                tracers,
                                                timestepper,
                                                free_surface,
                                                coriolis,
                                                forcing, 
                                                boundary_conditions=default_boundary_conditions)
end