using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, ExplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Grids: ZRegRectilinearGrid

function run_tracer_one_dimensional_immersed_diffusion(arch, underlying_grid, time_stepping)
    
    immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> 0))

    closure = IsotropicDiffusivity(κ = 1.0, time_discretization = time_stepping)

    model_kwargs = (tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields())

    model = HydrostaticFreeSurfaceModel(architecture = arch, grid = underlying_grid, 
                                                      closure = closure;
                                                      model_kwargs...)

    immersed_model = HydrostaticFreeSurfaceModel(architecture = arch, grid = immersed_grid,
                                                               closure = closure; 
                                                               model_kwargs...)
                                        
    initial_temperature(x, y, z) = exp(-z^2 / 0.02)
    [set!(m, c=initial_temperature) for m in (model, immersed_model)]

    diffusion_time_scale = model.grid.Δzᵃᵃᶜ^2 / model.closure.κ.c
    stop_time = 100diffusion_time_scale

    simulations = [simulation = Simulation(m, Δt = 1e-1 * diffusion_time_scale, stop_time = stop_time) for m in (model, immersed_model)]
    [run!(sim) for sim in simulations]

    test_domain = Int.((grid.Nz/2 + 1 + grid.Hz):(grid.Nz + grid.Hz))

    c          = Array(parent(model.tracers.c))[1, 1, test_domain]
    c_immersed = Array(parent(immersed_model.tracers.c))[1, 1, test_domain]

    @test all(c_immersed .≈ c)
end

function stretched_coord(N)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .- 0.5
    return z_faces
end

@inline show_coord(::ZRegRectilinearGrid) = "regular"
@inline show_coord(::RectilinearGrid)     = "stretched"

@testset "ImmersedVerticalDiffusion" begin
    for arch in archs
        
        time_steppings = (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
        
        N = 128
        z_stretched = stretched_coord(N)

        regular_grid   = RectilinearGrid(arch, size = N, z = (-0.5, 0.5), topology = (Flat, Flat, Bounded))
        stretched_grid = RectilinearGrid(arch, size = N, z = z_stretched, topology = (Flat, Flat, Bounded))
        
        for step in time_steppings, grid in (regular_grid, stretched_grid)
            @info "  Testing one-dimensional immersed diffusion [$(arch), $(step), $(show_coord(grid))]"
            run_tracer_one_dimensional_immersed_diffusion(arch, grid, step)
        end
    end
end