using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using SeawaterPolynomials.TEOS10
using Random

function ocean_benchmark(arch, Nx, Ny, Nz, topology, immersed, tracer_advection=WENO(order=7))    
    
    z_faces = collect(range(-6000, 0, length=Nz+1))

    grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), 
                                 halo=(7, 7, 7), 
                                    z=z_faces, 
                                    x=(-1000kilometers, 1000kilometers), 
                                    y=(-1000kilometers, 1000kilometers), 
                                    topology)

    grid = if immersed
        Random.seed!(1234)
        bottom = Oceananigans.Architectures.on_architecture(arch, - 5000 .* rand(Nx, Ny) .- 1000)
        ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map=true)
    else
        grid
    end
    
    @info "Grid is built"
    momentum_advection = WENOVectorInvariant()
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
    free_surface = SplitExplicitFreeSurface(grid; substeps=70)
    closure = CATKEVerticalDiffusivity()

    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection,
                                          tracer_advection,
                                          buoyancy,
                                          closure,
                                          free_surface,
                                          tracers = (:T, :S, :e))

    @info "Model is built"

    R = rand(size(model.grid))

    # initialize variables with randomish values
    Tᵢ = 1e-4 .* R .+ 20
    Sᵢ = 1e-4 .* R .+ 35
    uᵢ = 1e-6 .* R
    vᵢ = 1e-6 .* R
    
    set!(model, T=Tᵢ, S=Sᵢ, e=1e-6, u=uᵢ, v=vᵢ)

    return model
end

function run_benchmark(model)
    for _ in 1:15
        time_step!(model, 0.001)
    end
end

group = get(ENV, "BENCHMARK_GROUP", "all") |> Symbol

const Nx = 500
const Ny = 200
const Nz = 60

arch = GPU()

cheap_advection = FluxFormAdvection(WENO(order=7), WENO(order=7), Centered())

if group == :periodic
    model = ocean_benchmark(arch, Nx, Ny, Nz, (Periodic, Periodic, Bounded), false)
    run_benchmark(model)
end    

if group == :bounded
    model = ocean_benchmark(arch, Nx, Ny, Nz, (Bounded, Bounded, Bounded), false)
    run_benchmark(model)
end    

if group == :periodic_cheap_advection
    model = ocean_benchmark(arch, Nx, Ny, Nz, (Periodic, Periodic, Bounded), false, cheap_advection)
    run_benchmark(model)
end    

if group == :bounded_cheap_advection
    model = ocean_benchmark(arch, Nx, Ny, Nz, (Bounded, Bounded, Bounded), false, cheap_advection)
    run_benchmark(model)
end    

if group == :immersed
    model = ocean_benchmark(arch, Nx, Ny, Nz, (Periodic, Periodic, Bounded), true)
    run_benchmark(model)
end    