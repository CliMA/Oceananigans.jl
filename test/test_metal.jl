include("dependencies_for_runtests.jl")

using Metal

Oceananigans.defaults.FloatType = Float32

# Note that these tests are run on a virtualization framework
# via github actions runners and may break in the future.
# More about that:
# * https://github.com/CliMA/Oceananigans.jl/pull/4124#discussion_r1976449272
# * https://github.com/CliMA/Oceananigans.jl/pull/4152

#! format: off
@testset "MetalGPU extension" begin
    metal = Metal.MetalBackend()
    arch = GPU(metal)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa MtlArray
    @test parent(grid.xᶜᵃᵃ) isa MtlArray
    @test eltype(grid) == Float32
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(; grid,
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    @test parent(model.velocities.u) isa MtlArray
    @test parent(model.velocities.v) isa MtlArray
    @test parent(model.velocities.w) isa MtlArray
    @test parent(model.tracers.b) isa MtlArray

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes
end
#! format: on

@testset "MetalGPU: ImmersedBoundaryGrid" begin
    arch = GPU(Metal.MetalBackend())
    FT = Oceananigans.defaults.FloatType

    Lx, Ly, Lz = FT(5000meters), FT(5000meters), FT(20meters)
    Nx, Ny, Nz = 16, 16, 8

    underlying_grid = RectilinearGrid(
        arch;
        size=(Nx, Ny, Nz),
        x=(0, Lx),
        y=(0, Ly),
        z=(-Lz, 0),
        topology=(Bounded, Bounded, Bounded),
        halo=(5, 5, 5),
    )
    @test eltype(underlying_grid) == Float32

    @inline function depth(x::FT, y::FT)::FT
        a² = (FT(0.45) * Lx)^2  # semi-major squared axis
        b² = (FT(0.45) * Ly)^2  # semi-minor squared axis
        x₀ = FT(0.5) * Lx
        y₀ = FT(0.5) * Ly
        Δx = x - x₀
        Δy = y - y₀
        r² = (Δx * Δx) / a² + (Δy * Δy) / b²
        return r² ≤ FT(1) ? -Lz * (FT(1) - r²) : FT(1)
    end

    grid = ImmersedBoundaryGrid(
        underlying_grid,
        PartialCellBottom(depth);
        active_cells_map=false,
    )
    @test eltype(grid) == Float32

    Qᵀ = FT(0.01)
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ))

    Qᵘ = - FT(1e-4)
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵘ))

    model = HydrostaticFreeSurfaceModel(;
        grid,
        coriolis = FPlane(latitude=60),
        tracers = (:T, :S),
        buoyancy = SeawaterBuoyancy(),
        momentum_advection = WENO(),
        tracer_advection = WENO(),
        free_surface = SplitExplicitFreeSurface(grid; substeps=30), # default does not work on MetalGPU
        closure = nothing, # ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1, background_κz=1e-3),
        boundary_conditions = (u=u_bcs, T=T_bcs),
    )
    @test parent(model.velocities.u) isa MtlArray
    @test parent(model.velocities.v) isa MtlArray
    @test parent(model.velocities.w) isa MtlArray
    @test parent(model.tracers.T) isa MtlArray
    @test parent(model.tracers.S) isa MtlArray

    simulation = Simulation(model, Δt=5seconds, stop_iteration=20)
    run!(simulation)

    @test iteration(simulation) == 20
    @test time(simulation) == 100seconds
end

@testset "MetalGPU: TimeStepWizard" begin
    arch = GPU(Metal.MetalBackend())
    FT = Oceananigans.defaults.FloatType

    grid = RectilinearGrid(arch; size=(64, 64, 16), x=(0, 5000), y=(0, 5000), z=(-20, 0))
    @test eltype(grid) == Float32

    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENO(), tracer_advection=WENO(),
        free_surface=SplitExplicitFreeSurface(grid; substeps=30))
    sim = Simulation(model, Δt=5, stop_iteration=20)

    wizard = TimeStepWizard(cfl=FT(0.7), min_Δt=FT(1), max_Δt=FT(15))
    sim.callbacks[:wizard] = Callback(wizard, IterationInterval(5)) #TODO does not work with MetalGPU

    run!(sim)
    @test true
end
