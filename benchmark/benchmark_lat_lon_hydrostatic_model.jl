push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Oceananigans.Grids: metrics_precomputed

N = 256
                                
function multiple_steps!(model)
    for i = 1:20
        time_step!(model, 1e-6)
    end
    return nothing
end

for arch in [ has_cuda() ? [CPU(), GPU()] : [CPU()] ]

    grid_fly = LatitudeLongitudeGrid(size = (N, N, 1), 
                                     halo = (2, 2, 2), 
                                 latitude = (-60, 60), 
                                longitude = (-180, 180),
                                        z = (-10, 0),
                             architecture = arch)
    
    grid_pre = LatitudeLongitudeGrid(size = (N, N, 1), 
                                     halo = (2, 2, 2), 
                                 latitude = (-60, 60), 
                                longitude = (-180, 180),
                                        z = (-10, 0),
                             architecture = arch,
                       precompute_metrics = true)


    for grid in (grid_fly, grid_pre)

        model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = arch,    
                              momentum_advection = VectorInvariant(),
                                    free_surface = ExplicitFreeSurface())
                      
        time_step!(model, 1e-6) # warmup

        metrics_precomputed(grid) ? a = "precomputed metrics" : a = "calculated metrics"

        @info "Benchmarking $arch model with " * a * "..."
        @btime multiple_steps!($model)
    
    end
end
