using Reactant

using Test
using OffsetArrays
using Oceananigans
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.Utils: launch!
using SeawaterPolynomials: TEOS10EquationOfState
using KernelAbstractions: @kernel, @index
using Random

if haskey(ENV, "GPU_TEST")
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

OceananigansReactantExt = Base.get_extension(Oceananigans, :OceananigansReactantExt)
bottom_height(x, y) = - 0.5

function r_run!(sim, r_time_step!, r_first_time_step!)
    stop_iteration = sim.stop_iteration
    start_iteration = iteration(sim) + 1
    for n = start_iteration:stop_iteration
        if n == 1
            r_first_time_step!(sim.model, sim.Δt)
        else
            r_time_step!(sim.model, sim.Δt)
        end
    end
    return nothing
end

function test_reactant_model_correctness(GridType, ModelType, grid_kw, model_kw; immersed_boundary_grid=true)
    r_arch = ReactantState()
    r_grid = GridType(r_arch; grid_kw...)
    grid = GridType(CPU(); grid_kw...)

    if immersed_boundary_grid
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
        r_grid = ImmersedBoundaryGrid(r_grid, GridFittedBottom(bottom_height))
        @test isnothing(r_grid.interior_active_cells)
        @test isnothing(r_grid.active_z_columns)
        @test isnothing(grid.interior_active_cells)
        @test isnothing(grid.active_z_columns)
    end

    r_model = ModelType(; grid=r_grid, model_kw...)
    model = ModelType(; grid=grid, model_kw...)

    ui = randn(size(model.velocities.u)...)
    vi = randn(size(model.velocities.v)...)

    set!(model, u=ui, v=vi)
    set!(r_model, u=ui, v=vi)

    u, v, w = model.velocities
    ru, rv, rw = r_model.velocities

    # They will not be equal because r_model halos are not
    # filled during set!
    @test !(parent(u) ≈ parent(ru))
    @test !(parent(v) ≈ parent(rv))
    @test !(parent(w) ≈ parent(rw))

    Oceananigans.TimeSteppers.update_state!(r_model)

    # Test that fields were set correctly
    @info "  After setting an initial condition:"
    @show maximum(abs.(parent(u) .- parent(ru)))
    @show maximum(abs.(parent(v) .- parent(rv)))
    @show maximum(abs.(parent(w) .- parent(rw)))

    @test parent(u) ≈ parent(ru)
    @test parent(v) ≈ parent(rv)
    @test parent(w) ≈ parent(rw)

    # Deduce a stable time-step
    Δx = minimum_xspacing(grid)
    Δt = 0.01 * Δx

    # Stop iteration for both simulations
    stop_iteration = 3

    # What we normally do:
    simulation = Simulation(model; Δt, stop_iteration, verbose=false)
    run!(simulation)

    @info "  After running 3 time steps, the non-reactant model:"
    @test iteration(simulation) == stop_iteration
    @test time(simulation) == 3Δt

    # Reactant time now:
    r_simulation = Simulation(r_model; Δt, stop_iteration, verbose=false)

    Nsteps = ConcreteRNumber(3)
    @time "  Compiling r_run!:" begin
        r_first_time_step! = @compile sync=true Oceananigans.TimeSteppers.first_time_step!(r_model, Δt)
        r_time_step! = @compile sync=true Oceananigans.TimeSteppers.time_step!(r_model, Δt)
        r_time_step_sim! = @compile sync=true Oceananigans.TimeSteppers.time_step!(r_simulation)
    end

    @time "  Executing r_run!:" begin
        r_run!(r_simulation, r_time_step!, r_first_time_step!)
    end

    @info "  After running 3 time steps, the reactant model:"
    @test iteration(r_simulation) == stop_iteration
    @test time(r_simulation) == 3Δt

    # Some tests
    # Things ran normally:
    @test iteration(r_simulation) == iteration(simulation)
    @test time(r_simulation) == time(simulation)

    @show maximum(abs, parent(u))
    @show maximum(abs, parent(v))
    @show maximum(abs, parent(w))

    @show maximum(abs.(parent(u) .- parent(ru)))
    @show maximum(abs.(parent(v) .- parent(rv)))
    @show maximum(abs.(parent(w) .- parent(rw)))

    @test parent(u) ≈ parent(ru)
    @test parent(v) ≈ parent(rv)
    @test parent(w) ≈ parent(rw)

    # Running a few more time-steps works too:
    r_simulation.stop_iteration += 2
    r_run!(r_simulation, r_time_step!, r_first_time_step!)
    #r_time_step_for!(r_simulation, 2)
    @test iteration(r_simulation) == 5
    @test time(r_simulation) == 5Δt

    @test try
        r_time_step_sim!(r_simulation)
        true
    catch err
        false
        throw(err)
    end

    return r_simulation
end

function add_one!(f)
    arch = architecture(f)
    launch!(arch, f.grid, :xyz, _add_one!, f)
    return f
end

@kernel function _add_one!(f)
    i, j, k = @index(Global, NTuple)
    @inbounds f[i, j, k] += 1
end

