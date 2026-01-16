include("dependencies_for_runtests.jl")

using Metal

Oceananigans.defaults.FloatType = Float32

# Note that these tests are run on a virtualization framework
# via github actions runners and may break in the future.
# More about that:
# * https://github.com/CliMA/Oceananigans.jl/pull/4124#discussion_r1976449272
# * https://github.com/CliMA/Oceananigans.jl/pull/4152

@testset "MetalGPU extension" begin
    metal = Metal.MetalBackend()
    arch = GPU(metal)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa MtlArray
    @test parent(grid.xᶜᵃᵃ) isa MtlArray
    @test eltype(grid) == Float32
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(grid;
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

@testset "MetalGPU: ImmersedBoundaryGrid" begin
    arch = GPU(Metal.MetalBackend())
    FT = Float32

    Lx, Ly, Lz = FT(5000meters), FT(5000meters), FT(20meters)
    Nx, Ny, Nz = 16, 16, 8

    underlying_grid = RectilinearGrid(arch;
                                      size=(Nx, Ny, Nz),
                                      x=(0, Lx),
                                      y=(0, Ly),
                                      z=(-Lz, 0),
                                      topology=(Bounded, Bounded, Bounded),
                                      halo=(5, 5, 5))

    @test eltype(underlying_grid) == Float32

    @inline function depth(x, y)
        a² = (0.45*Lx)^2  # semi-major squared axis
        b² = (0.45*Ly)^2  # semi-minor squared axis
        x₀ = Lx / 2
        y₀ = Ly / 2
        Δx = x - x₀
        Δy = y - y₀
        r² = (Δx * Δx) / a² + (Δy * Δy) / b²
        return r² ≤ 1 ? -Lz * (1 - r²) : 1
    end

    grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(depth); active_cells_map=true)

    @test eltype(grid) == Float32

    Qᵀ = 1e-4
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ))

    Qᵘ = -1e-4
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵘ))

    model = HydrostaticFreeSurfaceModel(grid;
                                        coriolis=FPlane(latitude=60),
                                        tracers=(:T, :S),
                                        buoyancy=SeawaterBuoyancy(),
                                        momentum_advection=WENO(),
                                        tracer_advection=WENO(),
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=30), # default works on MetalGPU as well
                                        closure=ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1, background_κz=1e-3),
                                        boundary_conditions=(u=u_bcs, T=T_bcs))

    for field in merge(model.velocities, model.tracers)
        @test parent(field) isa MtlArray
    end

    simulation = Simulation(model, Δt=1, stop_iteration=10)
    run!(simulation)

    @test iteration(simulation) == 10
    @test time(simulation) == 10seconds
end

@testset "MetalGPU: test for reductions" begin
    arch = GPU(Metal.MetalBackend())
    grid = RectilinearGrid(arch, size=(32, 32, 32), extent=(1, 1, 1))

    # Test reduction of Field
    c = CenterField(grid)
    set!(c, 1)
    @test minimum(c) == 1

    # Test reduction of KernelFunctionOperation
    add_2(i, j, k, grid, c) = @inbounds c[i, j, k] + 2
    add_2_kfo = KernelFunctionOperation{Center,Center,Center}(add_2, grid, c)
    @test minimum(add_2_kfo) == 3
end

@testset "MetalGPU: TimeStepWizard" begin
    arch = GPU(Metal.MetalBackend())

    grid = RectilinearGrid(arch; size=(64, 64, 16), x=(0, 5000), y=(0, 5000), z=(-20, 0))
    @test eltype(grid) == Float32

    model = HydrostaticFreeSurfaceModel(grid; momentum_advection=WENO(), tracer_advection=WENO(),
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=30))

    sim = Simulation(model, Δt=5, stop_iteration=20)

    wizard = TimeStepWizard(cfl=0.7, min_Δt=1, max_Δt=15)
    sim.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

    run!(sim)
    @test time(sim) > 100seconds
end
