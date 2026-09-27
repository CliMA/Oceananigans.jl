# Setups shared by the tests and the validation of split tracer stepping with FluxFormSemiLagrangian slow tracers.
#
# Every model carries a fast buoyancy `b` (WENO5) that drives the dynamics, a fast uniform tracer `fast`, and three slow
# tracers: a random tracer `c` (inventory), a uniform tracer `constant`, and a smooth tracer `smooth` (errors).
# The slow tracers are passive, so the dynamics are identical for every ratio N and every slow advection scheme,
# and the dynamics time step Δt is fixed: the long-step tracer Courant number grows as N Δt.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.OrthogonalSphericalShellGrids: RightCenterFolded, RightFaceFolded
using Oceananigans.Models.HydrostaticFreeSurfaceModels: flux_form_semi_lagrangian_workspace
using Oceananigans.Architectures: on_architecture
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Vᶜᶜᶜ

include(joinpath(@__DIR__, "split_tracer_stepping_test_utils.jl"))

adaptive_implicit() = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)

# The vertical direction uses the same adaptive implicit WENO5 for every slow scheme, so that a failure
# at large N is caused by the horizontal scheme.
slow_advection(scheme::Symbol) = scheme == :FFSL ? FluxFormSemiLagrangian(vertical_scheme = WENO(order=5, time_discretization = adaptive_implicit())) :
                                                   WENO(order=5, time_discretization = adaptive_implicit())

fast_advection(scheme::Symbol) = scheme == :FFSL ? FluxFormSemiLagrangian() : WENO(order=5)

tracer_advection(slow, fast) = (b = WENO(order=5), fast = fast_advection(fast),
                                c = slow_advection(slow), constant = slow_advection(slow), smooth = slow_advection(slow))

ridge(x₀, width, height, depth) = (x, y) -> - depth + height * exp(- ((x - x₀) / width)^2)

function rectilinear_basin(arch = CPU(); halo = (6, 6, 4), size = (16, 8, 6))
    z = MutableVerticalDiscretization(collect(range(-60, 0, length = size[3] + 1)))
    underlying_grid = RectilinearGrid(arch; size, halo, x = (0, 32kilometers), y = (0, 16kilometers),
                                      z, topology = (Bounded, Bounded, Bounded))
    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge(16kilometers, 4kilometers, 30, 60)))
end

function tripolar_basin(arch = CPU(); fold_topology = RightCenterFolded, H = 100, halo = (7, 7, 4))
    z = MutableVerticalDiscretization(collect(range(-H, 0, length=4)))
    underlying_grid = TripolarGrid(arch; size = (20, 32, 3), halo, z, fold_topology)
    bump(λ, φ, λ₀) = exp(-(λ - λ₀)^2 / 50 - (φ - 55)^2 / 50)
    islands(λ, φ) = -H + 3H/2 * (bump(λ, φ, 70) + bump(λ, φ, 250) + bump(λ, φ, 430))
    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(islands))
end

split_configuration(ratio, slow_tracers = (:c, :constant, :smooth); kw...) =
    isnothing(ratio) ? nothing : TracerTimeStepSplitting(; tracers = slow_tracers, ratio, kw...)

"""
    baroclinic_basin_model(grid; ratio, slow = :FFSL, fast = :FFSL, buoyancy_contrast = 0.1)

A z-star baroclinic adjustment in a basin with an immersed ridge; the slumping front moves the free surface.
"""
function baroclinic_basin_model(grid; ratio, slow = :FFSL, fast = :FFSL, buoyancy_contrast = 0.1)
    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 8),
                                        tracer_advection = tracer_advection(slow, fast),
                                        vertical_coordinate = ZStarCoordinate(),
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :fast, :c, :constant, :smooth),
                                        tracer_time_step_splitting = split_configuration(ratio))

    x = first(nodes(grid, Center(), Center(), Center()))
    x₀ = (minimum(x) + maximum(x)) / 2
    bᵢ(x, y, z) = x < x₀ ? buoyancy_contrast : buoyancy_contrast / 5

    Random.seed!(1234)
    set!(model, b = bᵢ, fast = 1, c = (x, y, z) -> rand(), constant = 1, smooth = smooth_tracer_initial_condition(grid))

    return model
end

"""
    streamfunction_gyre_velocities(grid, speed)

Return CPU arrays with the interiors of `u` and `v` of a basin-scale gyre derived from a streamfunction `ψ` at the cell
corners of every level. `ψ` vanishes at every corner that touches an inactive (immersed or exterior) cell, so the
flow is discretely non-divergent level by level (in static cell volumes), goes around the immersed bathymetry, and
vanishes on immersed faces. `speed` is the maximum speed of the underlying smooth gyre; the boundary currents along
the immersed bathymetry are faster.
"""
function streamfunction_gyre_velocities(grid, speed)
    Nx, Ny, Nz = size(grid)
    active = active_cells(grid)
    underlying_grid = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Δx = minimum_xspacing(underlying_grid)
    Δy = minimum_yspacing(underlying_grid)
    Ψ = speed * Ny * Δy / π

    wet(i, j, k) = 1 ≤ i ≤ Nx && 1 ≤ j ≤ Ny && active[i, j, k]
    ψ = zeros(Nx + 1, Ny + 1, Nz)
    for k in 1:Nz, j in 1:Ny+1, i in 1:Nx+1
        surrounded = wet(i-1, j-1, k) && wet(i, j-1, k) && wet(i-1, j, k) && wet(i, j, k)
        ψ[i, j, k] = surrounded ? Ψ * sinpi((i - 1) / Nx) * sinpi((j - 1) / Ny) : 0
    end

    u = [- (ψ[i, j+1, k] - ψ[i, j, k]) / Δy for i in 1:Nx+1, j in 1:Ny, k in 1:Nz]
    v = [(ψ[i+1, j, k] - ψ[i, j, k]) / Δx for i in 1:Nx, j in 1:Ny+1, k in 1:Nz]

    return u, v
end

"""
    nondivergent_gyre_model(grid; ratio, slow = :FFSL, fast = :FFSL, speed = 0.35, bump_height = 1 / 10)

A z-star model in a `Bounded` rectilinear basin whose gyre goes around the immersed bathymetry (see
`streamfunction_gyre_velocities`). Without momentum advection and Coriolis the gyre is steady up to the
gravity waves radiated by the free surface bump, so the long-step Courant number is sustained while the horizontal
flow divergence (hence the net outflow of any cell during a long step) stays small.
"""
function nondivergent_gyre_model(grid; ratio, slow = :FFSL, fast = :FFSL, speed = 0.35, bump_height = 1 / 10)
    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 8),
                                        momentum_advection = nothing,
                                        tracer_advection = tracer_advection(slow, fast),
                                        vertical_coordinate = ZStarCoordinate(),
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :fast, :c, :constant, :smooth),
                                        tracer_time_step_splitting = split_configuration(ratio))

    x, y, _ = nodes(grid, Face(), Face(), Center())
    Lx = maximum(x) - minimum(x)
    Ly = maximum(y) - minimum(y)
    u, v = streamfunction_gyre_velocities(grid, speed)

    Random.seed!(1234)
    ηᵢ(x, y, z) = bump_height * exp(- ((x - Lx / 4)^2 + (y - Ly / 2)^2) / (Ly / 4)^2)
    smoothᵢ(x, y, z) = 1 + exp(- ((x - Lx / 4)^2 + (y - Ly / 2)^2) / (Ly / 3)^2)
    set!(model, u = u, v = v, η = ηᵢ, b = 0, fast = 1, c = (x, y, z) -> rand(), constant = 1, smooth = smoothᵢ)

    return model
end

"""
    gyre_basin_model(grid; ratio, slow = :FFSL, fast = :FFSL, speed = 0.9, bump_height = 1 / 10)

A z-star model in a basin with a basin-scale gyre that crosses the immersed ridge. Without momentum advection
and Coriolis the gyre persists (up to the gravity waves radiated where it crosses the ridge), so the long-step
Courant number is sustained and grows as `N Δt speed / Δx`.
"""
function gyre_basin_model(grid; ratio, slow = :FFSL, fast = :FFSL, speed = 0.9, bump_height = 1 / 10, substeps = 8)
    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps),
                                        momentum_advection = nothing,
                                        tracer_advection = tracer_advection(slow, fast),
                                        vertical_coordinate = ZStarCoordinate(),
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :fast, :c, :constant, :smooth),
                                        tracer_time_step_splitting = split_configuration(ratio))

    x, y, _ = nodes(grid, Face(), Face(), Center())
    Lx = maximum(x) - minimum(x)
    Ly = maximum(y) - minimum(y)
    Ψ = speed * Ly / π
    ψ = Field{Face, Face, Center}(grid)
    set!(ψ, (x, y, z) -> Ψ * sinpi(x / Lx) * sinpi(y / Ly))

    Random.seed!(1234)
    ηᵢ(x, y, z) = bump_height * exp(- ((x - Lx / 4)^2 + (y - Ly / 2)^2) / (Ly / 4)^2)
    smoothᵢ(x, y, z) = 1 + exp(- ((x - Lx / 4)^2 + (y - Ly / 2)^2) / (Ly / 5)^2)
    set!(model, u = ∂y(ψ), v = - ∂x(ψ), η = ηᵢ, b = 0, fast = 1, c = (x, y, z) -> rand(), constant = 1, smooth = smoothᵢ)

    return model
end

"""
    tripolar_rotation_model(grid; ratio, slow = :FFSL, fast = :FFSL, speed = 2)

A z-star model on a `TripolarGrid` whose (barotropic, unforced, non-rotating) flow crosses the fold,
and whose free surface starts from a bump. Without momentum advection and Coriolis the flow is steady
up to the gravity waves radiated by its divergence.
"""
function tripolar_rotation_model(grid; ratio, slow = :FFSL, fast = :FFSL, speed = 2, bump_height = 1 / 10)
    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                        momentum_advection = nothing,
                                        tracer_advection = tracer_advection(slow, fast),
                                        vertical_coordinate = ZStarCoordinate(),
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :fast, :c, :constant, :smooth),
                                        tracer_time_step_splitting = split_configuration(ratio))

    Random.seed!(2)
    uᵢ(λ, φ, z) = speed * sind(φ) * cosd(λ)
    vᵢ(λ, φ, z) = - speed * sind(λ)
    ηᵢ(λ, φ, z) = bump_height * exp(-((λ - 180)^2 + (φ - 60)^2) / 200)
    smoothᵢ(λ, φ, z) = 1 + exp(- ((λ - 60)^2 + (φ - 70)^2) / 800)
    set!(model, u = uᵢ, v = vᵢ, η = ηᵢ, b = 0, fast = 1,
         c = (λ, φ, z) -> 1 + cosd(φ) * cosd(λ) + rand() / 10, constant = 1, smooth = smoothᵢ)

    return model
end

#####
##### Diagnostics
#####

# Volume-based Courant numbers used by the FFSL step that was taken last (swept volume / upstream cell volume).
# `rows` restricts the diagnostic to the given j-rows of the x-faces and j-faces of the y-faces.
function swept_courant_numbers(model; rows = nothing)
    workspace = flux_form_semi_lagrangian_workspace(model.advection)
    isnothing(workspace) && return (NaN, NaN)
    sˣ = Array(interior(workspace.sˣ))
    sʸ = Array(interior(workspace.sʸ))
    if !isnothing(rows)
        sˣ = sˣ[:, filter(j -> j ≤ size(sˣ, 2), rows), :]
        sʸ = sʸ[:, filter(j -> j ≤ size(sʸ, 2), rows), :]
    end
    return maximum(abs, sˣ), maximum(abs, sʸ)
end

"""
    horizontal_outflow_courant_number(model, T)

Return the maximum over active cells of the net horizontal outflow during the most recent long step of length `T`,
`T ∇ₕ⋅(Aₕ ū) / V`, relative to the cell volume. Above one, the thickness left after the horizontal FFSL step of
that cell would be negative.
"""
function horizontal_outflow_courant_number(model, T)
    ū, v̄, _ = model.tracer_time_step_splitting.velocities
    grid = on_architecture(CPU(), model.grid)
    u = on_architecture(CPU(), ū)
    v = on_architecture(CPU(), v̄)
    active = active_cells(model.grid)
    Nx, Ny, Nz = size(grid)
    D = [T * flux_div_xyᶜᶜᶜ(i, j, k, grid, u, v) / Vᶜᶜᶜ(i, j, k, grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    return maximum(D[active])
end

