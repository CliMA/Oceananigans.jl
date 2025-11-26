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

@testset "MetalGPU: ImmersedBoundaryGrid" begin
    using Oceananigans
    using Oceananigans.Units
    using Oceananigans.BoundaryConditions: fill_halo_regions!
    using Metal

    Oceananigans.defaults.FloatType = Float32
    arch = GPU(Metal.MetalBackend())
    FT = Oceananigans.defaults.FloatType

    Qʰ = FT(-400.0)  # W m⁻², surface _heat_ flux (negative value - downward!)
    u10 = FT(5.0)    # Wind m/s

    Lx, Ly, Lz = FT(5000meters), FT(5000meters), FT(20meters)
    Nx, Ny, Nz = 128, 128, 32

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

    H = Array{FT}(undef, Nx, Ny)
    H .= FT(0.0)
    Δx = Lx / FT(Nx - 1)
    Δy = Ly / FT(Ny - 1)
    for i = 1:Nx
        x = (i - 1) * Δx
        for j = 1:Ny
            y = (j - 1) * Δy
            H[i, j] = depth(x, y)
        end
    end

    bathymetry = Field{Center,Center,Nothing}(underlying_grid)
    set!(bathymetry, coalesce.(H, FT(0.0)))
    fill_halo_regions!(bathymetry)

    grid = ImmersedBoundaryGrid(
        underlying_grid,
        # PartialCellBottom(depth; minimum_fractional_cell_height = FT(0.2)); #TODO does not work in the @testset env
        PartialCellBottom(bathymetry; minimum_fractional_cell_height=FT(0.2)); #TODO minimum_fractional_cell_height = FT(0.2) is required
        active_cells_map=false,
    )
    @test eltype(grid) == Float32

    cᴾ = FT(4184.0) # J K⁻¹ kg⁻¹, typical heat capacity for fresh water 
    ρₐ = FT(1.0)    # kg m⁻³, 
    ρ₀ = FT(1000.0) # kg m⁻³,
    Qᵀ = Qʰ / (ρ₀ * cᴾ) # K m s⁻¹, surface _temperature_ flux
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ))

    Cᴰ = FT(1.3e-3)
    τₐ = ρₐ * Cᴰ * u10 * abs(u10)
    Qᵘ = -τₐ / ρ₀ # m² s⁻²
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵘ))

    model = HydrostaticFreeSurfaceModel(;
        grid,
        coriolis=FPlane(latitude=60),
        tracers=(:T, :S),
        buoyancy=SeawaterBuoyancy(),
        momentum_advection=WENO(),
        tracer_advection=WENO(),
        free_surface=SplitExplicitFreeSurface(grid; substeps=30), # default does not work on MetalGPU
        closure=ConvectiveAdjustmentVerticalDiffusivity(convective_κz=FT(1.0), background_κz=FT(1e-3)),
        boundary_conditions=(u=u_bcs, T=T_bcs),
    )
    @test parent(model.velocities.u) isa MtlArray
    @test parent(model.velocities.v) isa MtlArray
    @test parent(model.velocities.w) isa MtlArray
    @test parent(model.tracers.T) isa MtlArray
    @test parent(model.tracers.S) isa MtlArray

    sim = Simulation(model, Δt=5seconds, stop_iteration=20)

    wizard = TimeStepWizard(cfl=FT(0.7), min_Δt=FT(1second), max_Δt=FT(15seconds))
    # sim.callbacks[:wizard] = Callback(wizard, IterationInterval(5)) #TODO does not work with MetalGPU

    run!(sim)

    @test iteration(sim) == 20
    @test time(sim) ≥ 100seconds
end

@testset "MetalGPU: TimeStepWizard" begin
    using Oceananigans
    using Metal

    #! format: off
    arch = GPU(Metal.MetalBackend()); Oceananigans.defaults.FloatType = Float32
    #! format: on
    # arch = CPU()
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
