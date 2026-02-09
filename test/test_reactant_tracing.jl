include("reactant_test_utils.jl")
using CUDA

# Test functions that exercise model tracing via @jit without triggering KA kernels.
# Each operates on the model struct, forcing Reactant to trace through
# NamedTuples, Fields, OffsetArrays, and grid constants.

sum_velocity(model) = sum(parent(model.velocities.u))

function fill_velocity!(model, val)
    parent(model.velocities.u) .= val
    return nothing
end

function scaled_sum(model)
    Lz = model.grid.Lz
    return Lz * sum(parent(model.velocities.u))
end

function field_difference(model)
    return sum(parent(model.velocities.u)) - sum(parent(model.velocities.v))
end

function tracer_sum(model)
    return sum(parent(model.tracers.b)) + sum(parent(model.tracers.c))
end

arch = ReactantState()

rectilinear_grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 2, 3))

lat_lon_grid = LatitudeLongitudeGrid(arch; size=(4, 4, 4),
                                     longitude=(0, 10), latitude=(0, 10), z=(-100, 0))

immersed_rectilinear_grid = ImmersedBoundaryGrid(
    RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1)),
    GridFittedBottom((x, y) -> -0.5))

immersed_lat_lon_grid = ImmersedBoundaryGrid(
    LatitudeLongitudeGrid(arch; size=(4, 4, 4),
                          longitude=(0, 10), latitude=(0, 10), z=(-100, 0)),
    GridFittedBottom((x, y) -> -50))

column_grid = RectilinearGrid(arch; size=8, z=(-10, 0), topology=(Flat, Flat, Bounded))

grids = [rectilinear_grid,
         lat_lon_grid,
         immersed_rectilinear_grid,
         immersed_lat_lon_grid,
         column_grid]

grid_names = ["RectilinearGrid",
              "LatitudeLongitudeGrid",
              "ImmersedBoundaryGrid{RectilinearGrid}",
              "ImmersedBoundaryGrid{LatitudeLongitudeGrid}",
              "single-column RectilinearGrid"]

for (grid, name) in zip(grids, grid_names)
    is_lat_lon = grid isa LatitudeLongitudeGrid ||
                 (grid isa ImmersedBoundaryGrid && grid.underlying_grid isa LatitudeLongitudeGrid)

    model_kwargs = is_lat_lon ?
        (momentum_advection=VectorInvariant(), free_surface=ExplicitFreeSurface()) :
        (free_surface=ExplicitFreeSurface(),)

    is_column = Oceananigans.Grids.topology(grid) == (Flat, Flat, Bounded)

    @testset "Reactant tracing [$name]" begin
        # Read: sum field data through traced OffsetArray
        model = HydrostaticFreeSurfaceModel(grid; buoyancy=nothing, tracers=(), model_kwargs...)
        ui = randn(size(model.velocities.u)...)
        set!(model, u=ui)
        @test @jit(sum_velocity(model)) ≈ sum(ui)

        if !is_column
            # Read: access two velocity fields and combine
            model = HydrostaticFreeSurfaceModel(grid; buoyancy=nothing, tracers=(), model_kwargs...)
            ui = randn(size(model.velocities.u)...)
            vi = randn(size(model.velocities.v)...)
            set!(model, u=ui, v=vi)
            @test @jit(field_difference(model)) ≈ sum(ui) - sum(vi)

            # Write: fill field data through traced OffsetArray
            model = HydrostaticFreeSurfaceModel(grid; buoyancy=nothing, tracers=(), model_kwargs...)
            @jit fill_velocity!(model, 3.0)
            @test all(Array(parent(model.velocities.u)) .≈ 3.0)
        end

        # Read: access grid scalar (Lz) as compile-time constant
        model = HydrostaticFreeSurfaceModel(grid; buoyancy=nothing, tracers=(), model_kwargs...)
        ui = randn(size(model.velocities.u)...)
        set!(model, u=ui)
        @test @jit(scaled_sum(model)) ≈ grid.Lz * sum(ui)

        # Read: trace through tracer fields
        model = HydrostaticFreeSurfaceModel(grid; buoyancy=BuoyancyTracer(), tracers=(:b, :c), model_kwargs...)
        bi = randn(size(model.tracers.b)...)
        ci = randn(size(model.tracers.c)...)
        set!(model, b=bi, c=ci)
        @test @jit(tracer_sum(model)) ≈ sum(bi) + sum(ci)
    end
end
