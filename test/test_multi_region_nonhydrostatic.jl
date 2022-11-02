using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: reconstruct_global_field, multi_region_object_from_array
using Oceananigans.Grids: min_Δx, min_Δy
using Oceananigans.Operators: hack_cosd

function Δ_min(grid) 
    Δx_min = min_Δx(grid)
    Δy_min = min_Δy(grid)
    return min(Δx_min, Δy_min)
end

# Simulation with random initial velocities
function random_nonhydrostatic_simulation(uᵢ, vᵢ, wᵢ, grid; P = XPartition, regions = 1, timestepper = :RungeKutta3)

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end
    mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices)

    model = NonhydrostaticModel(; grid = mrg, advection = WENO(), timestepper)

    u₀ = multi_region_object_from_array(uᵢ, mrg)
    v₀ = multi_region_object_from_array(vᵢ, mrg)
    w₀ = multi_region_object_from_array(wᵢ, mrg)

    set!(model, u=u₀, v=v₀, w=w₀)

    # Time-scale for advection across the smallest grid cell
    advection_time_scale = Δ_min(grid) / 0.5
    
    Δt = 0.1advection_time_scale
    
    for step in 1:10
        time_step!(model, Δt)
    end

    return model.velocities
end

Nx=16; Ny=16; Nz=16

for arch in archs

    regular_grid = RectilinearGrid(arch, size = (Nx, Ny, Nz),
                                   halo = (3, 3, 3),
                                   topology = (Periodic, Periodic, Periodic),
                                   x = (0, 1),
                                   y = (0, 1),
                                   z = (0, 1))


    vertically_unstretched_grid = RectilinearGrid(arch, size = (Nx, Ny, Nz),
                                                  halo = (3, 3, 3),
                                                  topology = (Periodic, Periodic, Bounded),
                                                  x = (0, 1),
                                                  y = (0, 1),
                                                  z = range(0, 1, length = Nz+1))

    partitioning = [XPartition, YPartition]

    uᵢ  = rand(Nx, Ny, Nz)
    vᵢ  = rand(Nx, Ny, Nz)
    wᵢᴿ = rand(Nx, Ny, Nz)
    wᵢᵁ = rand(Nx, Ny, Nz+1)

    @testset "Testing multi region tracer advection" begin
        for timestepper in [:QuasiAdamsBashforth2, :RungeKutta3], (grid, wᵢ) in zip([regular_grid, vertically_unstretched_grid], [wᵢᴿ, wᵢᵁ])
            uₑ, vₑ, wₑ = random_nonhydrostatic_simulation(uᵢ, vᵢ, wᵢ, grid; timestepper)

            uₑ = interior(uₑ)
            vₑ = interior(vₑ)
            wₑ = interior(wₑ)

            for regions in [2, 4], P in partitioning
                @info "  Testing $regions $(P)s with $(timestepper) on $(typeof(grid).name.wrapper) on the $arch"
                u, v, w = random_nonhydrostatic_simulation(uᵢ, vᵢ, wᵢ, grid; P=P, regions=regions, timestepper)

                u = interior(reconstruct_global_field(u))
                v = interior(reconstruct_global_field(v))
                w = interior(reconstruct_global_field(w))
            
                @test all(u .≈ uₑ)
                @test all(v .≈ vₑ)
                @test all(w .≈ wₑ)
            end
        end
    end
end
