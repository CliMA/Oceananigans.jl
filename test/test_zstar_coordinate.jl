include("dependencies_for_runtests.jl")

using Random
using Oceananigans: initialize!
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models: ZStarCoordinate, ZCoordinate

function test_zstar_coordinate(model, Ni, Δt, test_local_conservation=true)

    bᵢ = deepcopy(model.tracers.b)
    cᵢ = deepcopy(model.tracers.c)

    ∫bᵢ = Field(Integral(bᵢ))
    ∫cᵢ = Field(Integral(cᵢ))
    compute!(∫bᵢ)
    compute!(∫cᵢ)

    w   = model.velocities.w
    Nz  = model.grid.Nz

    for step in 1:Ni
        time_step!(model, Δt)

        ∫b = Field(Integral(model.tracers.b))
        ∫c = Field(Integral(model.tracers.c))
        compute!(∫b)
        compute!(∫c)

        condition = interior(∫b, 1, 1, 1) ≈ interior(∫bᵢ, 1, 1, 1)
        if !condition
            @info "Stopping early: buoyancy not conserved at step $step"
        end
        @test condition

        condition = interior(∫c, 1, 1, 1) ≈ interior(∫cᵢ, 1, 1, 1)
        if !condition
            @info "Stopping early: c tracer not conserved at step $step"
        end
        @test condition

        condition = maximum(abs, interior(w, :, :, Nz+1)) < eps(eltype(w))
        if !condition
            @info "Stopping early: nonzero vertical velocity at top at step $step"
        end
        @test condition

        # Constancy preservation test
        if test_local_conservation
            @test maximum(model.tracers.constant) ≈ 1
            @test minimum(model.tracers.constant) ≈ 1
        end
    end

    return nothing
end

const C = Center
const F = Face

@testset "MutableVerticalDiscretization tests" begin
    @info "testing the MutableVerticalDiscretization in ZCoordinate mode"

    z = MutableVerticalDiscretization((-20, 0))

    # A mutable immersed grid
    mutable_grid = RectilinearGrid(size=(2, 2, 20), x=(0, 2), y=(0, 1), z=z)
    mutable_grid = ImmersedBoundaryGrid(mutable_grid, GridFittedBottom((x, y) -> -10))

    # A static immersed grid
    static_grid = RectilinearGrid(size=(2, 2, 20), x=(0, 2), y=(0, 1), z=(-20, 0))
    static_grid = ImmersedBoundaryGrid(static_grid, GridFittedBottom((x, y) -> -10))

    # Make sure a model with a MutableVerticalDiscretization but ZCoordinate still runs and
    # the results are the same as a model with a static vertical discretization.
    kw = (; free_surface=ImplicitFreeSurface(), vertical_coordinate=ZCoordinate())
    mutable_model = HydrostaticFreeSurfaceModel(; grid=mutable_grid, kw...)
    static_model  = HydrostaticFreeSurfaceModel(; grid=static_grid, kw...)

    @test mutable_model.vertical_coordinate isa ZCoordinate
    @test static_model.vertical_coordinate isa ZCoordinate

    uᵢ = rand(size(mutable_model.velocities.u)...)
    vᵢ = rand(size(mutable_model.velocities.v)...)

    set!(mutable_model; u=uᵢ, v=vᵢ)
    set!(static_model;  u=uᵢ, v=vᵢ)

    static_sim  = Simulation(static_model;  Δt=1e-3, stop_iteration=100)
    mutable_sim = Simulation(mutable_model; Δt=1e-3, stop_iteration=100)

    run!(mutable_sim)
    run!(static_sim)

    # Check that fields are the same
    um, vm, wm = mutable_model.velocities
    us, vs, ws = static_model.velocities

    @test all(um.data .≈ us.data)
    @test all(vm.data .≈ vs.data)
    @test all(wm.data .≈ ws.data)
    @test all(um.data .≈ us.data)
end

@testset "ZStarCoordinate diffusion test" begin
    Random.seed!(1234)

    # Build a stretched vertical coordinate
    z_static = [i + rand() for i in -15:0]
    z_static[1] = -15
    z_static[end] = 0
    z_moving = MutableVerticalDiscretization(z_static ./ 1.5)

    for arch in archs
        c₀ = rand(15)

        grid_static = RectilinearGrid(arch; size=(1, 1, 15), x=(0, 1), y=(0, 1), z=z_static, topology=(Periodic, Periodic, Bounded))
        grid_moving = RectilinearGrid(arch; size=(1, 1, 15), x=(0, 1), y=(0, 1), z=z_moving, topology=(Periodic, Periodic, Bounded))

        fill!(grid_moving.z.ηⁿ,   5)
        fill!(grid_moving.z.σᶜᶜ⁻, 1.5)
        fill!(grid_moving.z.σᶜᶜⁿ, 1.5)
        fill!(grid_moving.z.σᶜᶠⁿ, 1.5)
        fill!(grid_moving.z.σᶠᶠⁿ, 1.5)
        fill!(grid_moving.z.σᶠᶜⁿ, 1.5)

        for TD in (ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization)
            for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3, :SplitRungeKutta5) #timesteppers
                for c_bcs in (FluxBoundaryCondition(nothing), FluxBoundaryCondition(0.01), ValueBoundaryCondition(0.01))
                    @info "testing ZStarCoordinate diffusion on $(typeof(arch)) with $TD, $timestepper, and $c_bcs at the top"

                    model_static = HydrostaticFreeSurfaceModel(; grid = grid_static,
                                                                tracers = :c,
                                                                timestepper,
                                                                boundary_conditions = (; c = FieldBoundaryConditions(top=c_bcs)),
                                                                closure = VerticalScalarDiffusivity(TD(), κ=0.1))

                    model_moving = HydrostaticFreeSurfaceModel(; grid = grid_moving,
                                                                tracers = :c,
                                                                timestepper,
                                                                boundary_conditions = (; c = FieldBoundaryConditions(top=c_bcs)),
                                                                closure = VerticalScalarDiffusivity(TD(), κ=0.1))

                    set!(model_static, c=c₀)
                    set!(model_moving, c=c₀, η=5)

                    for _ in 1:1000
                        time_step!(model_static, 1.0)
                        time_step!(model_moving, 1.0)
                    end

                    @test all(Array(interior(model_static.tracers.c)) .≈ Array(interior(model_moving.tracers.c)))
                end
            end
        end
    end
end
