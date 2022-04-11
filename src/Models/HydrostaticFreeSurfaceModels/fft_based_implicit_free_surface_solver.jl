using Oceananigans.Grids
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Grids: x_domain, y_domain
using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Fields: ReducedField
using Statistics

import Oceananigans.Solvers: solve!

struct FFTImplicitFreeSurfaceSolver{S, G3, G2, R}
    fft_poisson_solver :: S
    three_dimensional_grid :: G3
    horizontal_grid :: G2
    right_hand_side :: R
end

validate_fft_implicit_solver_grid(grid) = 
    grid isa RegRectilinearGrid || grid isa HRegRectilinearGrid ||
        throw(ArgumentError("FFTImplicitFreeSurfaceSolver requires horizontally-regular rectilinear grids."))

validate_fft_implicit_solver_grid(ibg::ImmersedBoundaryGrid) =
    validate_fft_implicit_solver_grid(ibg.grid)

"""
    FFTImplicitFreeSurfaceSolver(grid, settings=nothing, gravitational_acceleration=nothing)

Return a solver based on the fast Fourier transform for the elliptic equation
    
```math
[∇² - 1 / (g H Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - ηⁿ / Δt) / (g H Δt)
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with constant depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at time-step `n`, `ηⁿ`.
"""
function FFTImplicitFreeSurfaceSolver(grid, settings=nothing, gravitational_acceleration=nothing)

    validate_fft_implicit_solver_grid(grid)

    # Construct a "horizontal grid". We support either x or y being Flat, but not both.
    TX, TY, TZ = topology(grid)
    sz = Nx, Ny = (grid.Nx, grid.Ny)
    halo = (grid.Hx, grid.Hy)
    domain = (x = x_domain(grid), y = y_domain(grid))

    # Reduce kwargs.
    # Either [1, 2], [1], or [2]
    nonflat_dims = findall(T -> !(T() isa Flat), (TX, TY))

    sz = Tuple(sz[i] for i in nonflat_dims)
    halo = Tuple(halo[i] for i in nonflat_dims)
    domain = NamedTuple((:x, :y)[i] => domain[i] for i in nonflat_dims)

    # Build a "horizontal grid" with a Flat vertical direction.
    # Even if the three dimensional grid is vertically stretched, we can only use
    # FFTImplicitFreeSurfaceSolver with grids that are regularly spaced in the
    # horizontal direction.
    horizontal_grid = RectilinearGrid(architecture(grid);
                                      topology = (TX, TY, Flat),
                                      size = sz,
                                      halo = halo,
                                      domain...)

    solver = FFTBasedPoissonSolver(horizontal_grid)
    right_hand_side = solver.storage

    return FFTImplicitFreeSurfaceSolver(solver, grid, horizontal_grid, right_hand_side)
end

build_implicit_step_solver(::Val{:FastFourierTransform}, grid, settings, gravitational_acceleration) =
    FFTImplicitFreeSurfaceSolver(grid, settings, gravitational_acceleration)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::FFTImplicitFreeSurfaceSolver, rhs, g, Δt)
    solver = implicit_free_surface_solver.fft_poisson_solver
    grid = implicit_free_surface_solver.three_dimensional_grid
    Lz = grid.Lz

    # LHS constant
    m = - 1 / (g * Lz * Δt^2) # units L⁻²

    # solve! is blocking:
    solve!(η, solver, rhs, m)

    return η
end

function compute_implicit_free_surface_right_hand_side!(rhs, implicit_solver::FFTImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    poisson_solver = implicit_solver.fft_poisson_solver
    arch = architecture(poisson_solver)
    grid = implicit_solver.three_dimensional_grid
    Lz = grid.Lz

    event = launch!(arch, grid, :xy,
                    fft_implicit_free_surface_right_hand_side!,
                    rhs, grid, g, Lz, Δt, ∫ᶻQ, η,
                    dependencies = device_event(arch))
    
     wait(device(arch), event)
    return nothing
end

@kernel function fft_implicit_free_surface_right_hand_side!(rhs, grid, g, Lz, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᶜ(i, j, 1, grid)
    δ_Q = flux_div_xyᶜᶜᶜ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    @inbounds rhs[i, j, 1] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * Lz * Δt * Az)
end

