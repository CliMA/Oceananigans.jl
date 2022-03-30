using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Statistics
using Printf
using Oceananigans.MultiRegion
using LinearAlgebra, SparseArrays
using Oceananigans.Solvers: constructors, unpack_constructors
using Oceananigans.Grids: architecture
using Oceananigans.Utils

function geostrophic_adjustment_test(free_surface, grid; regions = 1)

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end
    mrg = MultiRegionGrid(grid, partition = XPartition(regions), devices = devices)

    coriolis = FPlane(f = 1e-4)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        coriolis = coriolis,
                                        free_surface = free_surface)

    gaussian(x, L) = exp(-x^2 / 2L^2)

    U = 0.1 # geostrophic velocity
    L  = grid.Lx / 40 # gaussian width
    x₀ = grid.Lx / 4 # gaussian center

    vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

    g = model.free_surface.gravitational_acceleration
    η = model.free_surface.η

    η₀ = coriolis.f * U * L / g # geostrohpic free surface amplitude

    ηᵍ(x) = η₀ * gaussian(x - x₀, L)

    ηⁱ(x, y) = 2 * ηᵍ(x)

    set!(model, v = vᵍ)
    @apply_regionally set!(η, ηⁱ)

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    Δt = 2 * model.grid.Δxᶜᵃᵃ / gravity_wave_speed

    for step in 1:10
        time_step!(model, Δt)
    end

    return η
end

Lh = 100kilometers
Lz = 400meters

for arch in [GPU()] #archs

    free_surface   = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient, maximum_iterations = 64 * 3)
    topology_types = (Bounded, Periodic, Bounded), (Periodic, Periodic, Bounded)

    @testset "Testing multi region implicit free surface" begin
        for topology_type in topology_types
            grid = RectilinearGrid(arch,
                        size = (64, 3, 1),
                        x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
                        topology = topology_type)

            ηs = geostrophic_adjustment_test(free_surface, grid);
            ηs = interior(ηs);

            for regions in [2, 4]
                @info "  Testing $regions partitions on $(topology_type) on the $arch"
                η = geostrophic_adjustment_test(free_surface, grid, regions = regions)
                
                η = construct_regionally(interior, η)
                
                for region in 1:regions
                    init = Int(size(ηs, 1) / regions) * (region - 1) + 1
                    fin  = Int(size(ηs, 1) / regions) * region
                    @test all(Array(η[region]) .≈ Array(ηs)[init:fin, :, :])
                end
            end
        end
    end
end