push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using CUDA
using Oceananigans
using Benchmarks

N = 256

for arch in [ has_cuda() ? [CPU, GPU] : [CPU] ]

    grid_pre = LatitudeLongitudeGrid(size = (N, N, 3), halo = (2, 2, 2), latitude = (-160, 160), longitude = (-180, 180), z = (-10, 0))
    grid_fly = LatitudeLongitudeGrid(size = (N, N, 3), halo = (2, 2, 2), latitude = (-160, 160), longitude = (-180, 180), z = (-10, 0))
                                
    function ten_steps!(model)
        for _ = 1:10
            time_step!(model, 1e-6)
        end
        return nothing
    end

    for arch in (CPU(), GPU())

        for grid in (grid_fly, grid_pre)

model = HydrostaticFreeSurfaceModel(grid = grid,
                            architecture = arch,    
                      momentum_advection = VectorInvariant(),
                            free_surface = ExplicitFreeSurface())
          
          
            
            time_step!(model, 1e-6) # warmup

            @info "Benchmarking $arch model with $(short_show(grid))..."
            @btime ten_steps!($model)
        end
    end

end