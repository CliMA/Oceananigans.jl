using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Models: AbstractModel
using Oceananigans.DistributedComputations
using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, update_state!, rk_substep!, tick!, cache_current_fields!
using MPI
using Test
MPI.Init()

if MPI.Comm_size(MPI.COMM_WORLD) == 1
    # This works with any free surface except `ImplicitFreeSurface`
    arch = CPU()
else
    # This breaks local conservation with the `SplitExplicitFreeSurface`
    # arch = Distributed(CPU(), partition = Partition(1, 4), synchronized_communication=true)
    # This works with any free surface except `ImplicitFreeSurface`
    # arch = Distributed(CPU(), partition = Partition(4, 1), synchronized_communication=true)
end

z_stretched = MutableVerticalDiscretization(collect(-2:0))
grid = RectilinearGrid(arch; size = (2, 2, 2), 
                                x = (0, 2), 
                                y = (0, 2), 
                                topology = (Periodic, Bounded, Bounded), 
                                z = z_stretched)


grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -1))

# These the free surfaces do not work in combination
# with  arch = Distributed(CPU(), partition = Partition(1, 4), synchronized_communication=true)
# free_surface = SplitExplicitFreeSurface(grid; substeps=20)
# This free surfaces work correctly with any architecture
# free_surface = ExplicitFreeSurface()
free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=nothing, reltol=1e-15) # SplitExplicitFreeSurface(grid; substeps=20) #
timestepper = Oceananigans.TimeSteppers.SplitRungeKuttaTimeStepper(coefficients = (3, 2, 1, ))

model = HydrostaticFreeSurfaceModel(grid;
                                    free_surface,
                                    tracers = (:b, :c, :constant),
                                    timestepper,
                                    momentum_advection = nothing,
                                    buoyancy = BuoyancyTracer(),
                                    vertical_coordinate = ZStarCoordinate())

bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01
set!(model, c = (x, y, z) -> rand(), b = bᵢ, constant = 1)

Δt = 1
b₁ = deepcopy(model.tracers.b)
c₁ = deepcopy(model.tracers.c)

∫b₁ = Field(Integral(b₁))
∫c₁ = Field(Integral(c₁))
compute!(∫b₁)
compute!(∫c₁)

w   = model.velocities.w
Nz  = model.grid.Nz

callbacks = []

function my_time_step!(model::AbstractModel{<:SplitRungeKuttaTimeStepper}, Δt; callbacks=[])

    if model.clock.iteration == 0
        update_state!(model, callbacks)
    end

    cache_current_fields!(model)
    grid = model.grid

    ####
    #### Loop over the stages
    ####

    for (stage, β) in enumerate(model.timestepper.β)
        # Update the clock stage
        model.clock.stage = stage

        # Perform the substep
        Δτ = Δt / β
        rk_substep!(model, Δτ, callbacks)

        # Update the state
        update_state!(model, callbacks)
        check_conditions(model)
    end

    # Finalize step
    tick!(model.clock, Δt)

    return nothing
end

# Check conservation, conservation should work no problem!!
function check_conditions(model)
    ∫b = Field(Integral(model.tracers.b))
    ∫c = Field(Integral(model.tracers.c))
    compute!(∫b)
    compute!(∫c)
    condition = interior(∫b, 1, 1, 1) ≈ interior(∫b₁, 1, 1, 1)
    if !condition
        @info "Stopping early: buoyancy not conserved at step $step"
    end
    @test condition

    condition = interior(∫c, 1, 1, 1) ≈ interior(∫c₁, 1, 1, 1)
    if !condition
        @info "Stopping early: c tracer not conserved at step $step"
    end
    @test condition

    condition = maximum(abs, interior(w, :, :, Nz+1)) < eps(eltype(w))
    if !condition
        @info "Stopping early: nonzero vertical velocity at top at step $step"
    end
    @test condition

    # # Constancy preservation test
    @test maximum(model.tracers.constant) ≈ 1
    @test minimum(model.tracers.constant) ≈ 1
end

for step in 1:100
    @root @info "Testing step $step"
    my_time_step!(model, Δt)
    check_conditions(model)
end
