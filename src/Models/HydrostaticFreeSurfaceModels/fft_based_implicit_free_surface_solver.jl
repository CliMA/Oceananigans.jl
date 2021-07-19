using Oceananigans.Grids
using Oceananigans.Grids: all_x_nodes, all_y_nodes
using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Fields: ReducedField
using Statistics

import Oceananigans.Solvers: solve!

struct FFTBasedImplicitFreeSurfaceSolver{S, G3, G2, R}
    fft_based_poisson_solver :: S
    three_dimensional_grid :: G3
    horizontal_grid :: G2
    right_hand_side :: R
end

"""
```math
(∇² - Az / (g H Δt²)) ηⁿ⁺¹ = 1 / (g H Δt) * (∇ʰ ⋅ Q★ - Az ηⁿ / Δt)
```
"""
function FFTBasedImplicitFreeSurfaceSolver(arch::AbstractArchitecture, grid, settings)

    grid isa RegularRectilinearGrid || grid isa VerticallyStretchedRectilinearGrid ||
        throw(ArgumentError("FFTBasedImplicitFreeSurfaceSolver requires horizontally-regular rectilinear grids."))

    topo = topology(grid)
    x = all_x_nodes(Face, grid)
    y = all_y_nodes(Face, grid)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
    
    horizontal_grid = RegularRectilinearGrid(topology = (topo[1], topo[2], Flat),
                                             size = (grid.Nx, grid.Ny),
                                             x = (x[1], x[Nx+1]),
                                             y = (y[1], y[Ny+1]),
                                             halo = (Hx, Hy))

    solver = FFTBasedPoissonSolver(arch, horizontal_grid)
    right_hand_side = solver.storage

    return FFTBasedImplicitFreeSurfaceSolver(solver, grid, horizontal_grid, right_hand_side)
end

build_implicit_step_solver(::Val{:FFTBased}, arch, grid, settings) =
    FFTBasedImplicitFreeSurfaceSolver(arch, grid, settings)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::FFTBasedImplicitFreeSurfaceSolver, rhs, g, Δt)
    solver = implicit_free_surface_solver.fft_based_poisson_solver
    grid = implicit_free_surface_solver.three_dimensional_grid
    H = grid.Lz

    m = - 1 / (g * H * Δt^2) # units L⁻²

    # solve! is blocking:
    solve!(η, solver, rhs, m)

    fill_halo_regions!(η, solver.architecture)
    
    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs,
                                                        implicit_solver::FFTBasedImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.fft_based_poisson_solver
    arch = solver.architecture
    grid = implicit_solver.three_dimensional_grid
    H = grid.Lz

    event = launch!(arch, grid, :xy,
                    fft_based_implicit_free_surface_right_hand_side!,
                    rhs, grid, g, H, Δt, ∫ᶻQ, η)

    return event
end

@kernel function fft_based_implicit_free_surface_right_hand_side!(rhs, grid, g, H, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᵃ(i, j, 1, grid)
    δ_Q = flux_div_xyᶜᶜᵃ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    @inbounds rhs[i, j, 1] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * H * Δt * Az)
end
