include("dependencies_for_runtests.jl")

using Oceananigans.Grids: XDirection, YDirection, ZDirection
using Oceananigans.Models: BulkDrag, BulkDragFunction, BulkDragBoundaryCondition,
                           QuadraticFormulation, LinearFormulation
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions, IMEXFluxTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, ImmersedBoundaryCondition

@testset "BulkDrag" begin
    @info "Testing BulkDrag..."

    #####
    ##### Test BulkDragFunction construction
    #####

    @testset "BulkDragFunction construction" begin
        # Default construction (QuadraticFormulation)
        df = BulkDragFunction(coefficient=1e-3)
        @test isnothing(df.direction)
        @test df.formulation isa QuadraticFormulation
        @test df.coefficient == 1e-3
        @test df.background_velocities == (0, 0, 0)

        # With direction
        df_x = BulkDragFunction(direction=XDirection(), coefficient=1e-3)
        @test df_x.direction isa XDirection
        @test df_x.coefficient == 1e-3

        df_y = BulkDragFunction(direction=YDirection(), coefficient=1e-3)
        @test df_y.direction isa YDirection

        # Custom coefficient
        df = BulkDragFunction(coefficient=0.003)
        @test df.coefficient == 0.003

        # LinearFormulation
        df_linear = BulkDragFunction(LinearFormulation(), coefficient=0.01)
        @test df_linear.formulation isa LinearFormulation

        # Custom background velocities (3D)
        df = BulkDragFunction(coefficient=1e-3, background_velocities=(0.1, 0.2, 0.0))
        @test df.background_velocities == (0.1, 0.2, 0.0)

        # Test show method
        @test occursin("BulkDragFunction", sprint(show, df))
        @test occursin("QuadraticFormulation", sprint(show, df))
    end

    #####
    ##### Test BulkDrag boundary condition
    #####

    @testset "BulkDrag boundary condition" begin
        u_drag = BulkDrag(direction=XDirection(), coefficient=1e-3)
        v_drag = BulkDrag(direction=YDirection(), coefficient=1e-3)

        @test u_drag isa BulkDragBoundaryCondition
        @test v_drag isa BulkDragBoundaryCondition

        @test u_drag.condition.direction isa XDirection
        @test v_drag.condition.direction isa YDirection
        @test u_drag.condition.formulation isa QuadraticFormulation
    end

    #####
    ##### Test quadratic drag tendency on bottom boundary
    #####

    @testset "Quadratic drag tendency on bottom boundary" begin
        for arch in archs
            # Use a unit grid for simple math: Az = Δx*Δy = 1, V = Δx*Δy*Δz = 1
            grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

            Cᴰ = 0.1
            U₀ = 0.5
            drag_bc = BulkDrag(QuadraticFormulation(), coefficient=Cᴰ)

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs,))

            # Check direction was inferred correctly
            @test model.velocities.u.boundary_conditions.bottom.condition.direction isa XDirection
            @test model.velocities.u.boundary_conditions.bottom.condition.formulation isa QuadraticFormulation

            # Set initial velocity field (u=U₀, v=w=0)
            set!(model, u=U₀)

            # Time step with small Δt and verify velocity change
            Δt = 1e-6
            time_step!(model, Δt)

            # Expected quadratic drag: du/dt = -Cᴰ |U| u = -Cᴰ U₀² (since v=w=0, |U|=U₀)
            # After one step: Δu ≈ -Cᴰ U₀² * Δt
            expected_Δu = -Cᴰ * U₀^2 * Δt
            actual_Δu = Array(interior(model.velocities.u))[1, 1, 1] - U₀

            @test actual_Δu ≈ expected_Δu rtol=1e-4
        end
    end

    #####
    ##### Test that drag opposes the flow on right boundaries (top, east, north), where a flux enters the tendency as -J/Δ
    #####

    @testset "Quadratic drag tendency on top boundary" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

            Cᴰ = 0.1
            U₀ = 0.5
            u_bcs = FieldBoundaryConditions(top=BulkDrag(coefficient=Cᴰ))
            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs,))
            set!(model, u=U₀)

            Δt = 1e-6
            time_step!(model, Δt)

            expected_Δu = -Cᴰ * U₀^2 * Δt
            actual_Δu = Array(interior(model.velocities.u))[1, 1, 1] - U₀

            @test actual_Δu ≈ expected_Δu rtol=1e-4
        end
    end

    @testset "BulkDrag decelerates the flow on every domain boundary" begin
        for arch in archs
            Cᴰ = 0.1
            U₀ = 0.1
            zgrid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
            xgrid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1), topology=(Bounded, Periodic, Periodic))
            ygrid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Bounded, Periodic))
            drag = BulkDrag(coefficient=Cᴰ)

            cases = ((zgrid, :u, :bottom), (zgrid, :u, :top),
                     (xgrid, :v, :west),   (xgrid, :v, :east),
                     (ygrid, :u, :south),  (ygrid, :u, :north))

            for (grid, name, side) in cases
                bcs = NamedTuple{(name,)}((FieldBoundaryConditions(; side => drag),))
                model = NonhydrostaticModel(grid; boundary_conditions=bcs, advection=nothing, closure=nothing)
                set!(model; name => U₀)
                for _ in 1:10
                    time_step!(model, 0.1)
                end
                Ū = sum(Array(interior(model.velocities[name]))) / length(interior(model.velocities[name]))
                @test 0 < Ū < U₀
            end
        end
    end

    #####
    ##### Test linear drag tendency on bottom boundary
    #####

    @testset "Linear drag tendency on bottom boundary" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

            Cᴰ = 0.1
            U₀ = 0.5
            drag_bc = BulkDrag(LinearFormulation(), coefficient=Cᴰ)

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs,))

            @test model.velocities.u.boundary_conditions.bottom.condition.formulation isa LinearFormulation

            set!(model, u=U₀)

            Δt = 1e-6
            time_step!(model, Δt)

            # Expected linear drag: du/dt = -Cᴰ u = -Cᴰ U₀
            # After one step: Δu ≈ -Cᴰ U₀ * Δt
            expected_Δu = -Cᴰ * U₀ * Δt
            actual_Δu = Array(interior(model.velocities.u))[1, 1, 1] - U₀

            @test actual_Δu ≈ expected_Δu rtol=1e-4
        end
    end

    #####
    ##### Test automatic direction inference with NonhydrostaticModel
    #####

    @testset "Automatic direction inference with NonhydrostaticModel" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))

            Cᴰ = 0.003
            drag_bc = BulkDrag(coefficient=Cᴰ)

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            v_bcs = FieldBoundaryConditions(bottom=drag_bc)

            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs, v=v_bcs))

            # Check that direction was inferred correctly
            @test model.velocities.u.boundary_conditions.bottom.condition.direction isa XDirection
            @test model.velocities.v.boundary_conditions.bottom.condition.direction isa YDirection

            set!(model, u=0.1, v=0.1)

            # Time step and verify model advances
            time_step!(model, 1e-3)
            time_step!(model, 1e-3)
            time_step!(model, 1e-3)

            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test with background velocities
    #####

    @testset "BulkDrag with background velocities" begin
        for arch in archs
            # Unit grid for simple tendency verification
            grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

            Cᴰ = 0.1
            U∞ = 0.0
            V∞ = 0.2  # Background along-y velocity
            W∞ = 0.0

            drag_bc = BulkDrag(coefficient=Cᴰ, background_velocities=(U∞, V∞, W∞))

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs,))

            # Check background velocities are stored correctly
            @test model.velocities.u.boundary_conditions.bottom.condition.background_velocities == (U∞, V∞, W∞)

            # Set u=U₀ and verify the combined effect with background velocity
            U₀ = 0.3
            set!(model, u=U₀)

            Δt = 1e-6
            time_step!(model, Δt)

            # Speed: |U + U∞| = √((U₀ + U∞)² + V∞²) = √(U₀² + V∞²)
            speed = sqrt(U₀^2 + V∞^2)
            expected_Δu = -Cᴰ * speed * U₀ * Δt
            actual_Δu = Array(interior(model.velocities.u))[1, 1, 1] - U₀

            @test actual_Δu ≈ expected_Δu rtol=1e-4
        end
    end

    #####
    ##### Test with HydrostaticFreeSurfaceModel
    #####

    @testset "BulkDrag with HydrostaticFreeSurfaceModel" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))

            Cᴰ = 0.003
            drag_bc = BulkDrag(coefficient=Cᴰ)

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            v_bcs = FieldBoundaryConditions(bottom=drag_bc)

            model = HydrostaticFreeSurfaceModel(grid; boundary_conditions=(u=u_bcs, v=v_bcs))

            set!(model, u=0.1, v=0.1)

            time_step!(model, 1e-3)
            time_step!(model, 1e-3)
            time_step!(model, 1e-3)

            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test with ImmersedBoundaryGrid
    #####

    @testset "BulkDrag with ImmersedBoundaryGrid" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))

            bottom(x, y) = -0.5
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

            Cᴰ = 0.003
            drag_bc = BulkDrag(coefficient=Cᴰ)

            u_bcs = FieldBoundaryConditions(bottom=drag_bc, immersed=ImmersedBoundaryCondition(bottom=drag_bc))
            v_bcs = FieldBoundaryConditions(bottom=drag_bc, immersed=ImmersedBoundaryCondition(bottom=drag_bc))

            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs, v=v_bcs))

            set!(model, u=0.1, v=0.1)

            time_step!(model, 1e-3)
            time_step!(model, 1e-3)
            time_step!(model, 1e-3)

            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test with ImmersedBoundaryCondition
    #####

    @testset "BulkDrag with ImmersedBoundaryCondition" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))

            bottom(x, y) = -0.5
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

            Cᴰ = 0.003
            drag_bc = BulkDrag(coefficient=Cᴰ)

            u_ibc = ImmersedBoundaryCondition(bottom=drag_bc)
            v_ibc = ImmersedBoundaryCondition(bottom=drag_bc)

            u_bcs = FieldBoundaryConditions(immersed=u_ibc)
            v_bcs = FieldBoundaryConditions(immersed=v_ibc)

            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs, v=v_bcs))

            set!(model, u=0.1, v=0.1)

            time_step!(model, 1e-3)
            time_step!(model, 1e-3)
            time_step!(model, 1e-3)

            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test drag on lateral boundaries
    #####

    @testset "BulkDrag on lateral boundaries" begin
        for arch in archs
            # Test that BulkDrag works on lateral (x-normal) boundaries
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1),
                                   topology=(Bounded, Bounded, Bounded))

            Cᴰ = 0.1
            drag_bc = BulkDrag(coefficient=Cᴰ)

            # v-velocity can have drag on west/east (x-normal) boundaries
            v_bcs = FieldBoundaryConditions(west=drag_bc, east=drag_bc)
            model = NonhydrostaticModel(grid; boundary_conditions=(v=v_bcs,))

            @test model.velocities.v.boundary_conditions.west.condition.direction isa YDirection
            @test model.velocities.v.boundary_conditions.east.condition.direction isa YDirection

            set!(model, v=0.1)

            # Verify model runs forward
            time_step!(model, 1e-3)
            time_step!(model, 1e-3)
            time_step!(model, 1e-3)

            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Implicit-explicit time discretization
    #####

    @testset "IMEX BulkDrag" begin
        imex = IMEXFluxTimeDiscretization()

        @test_throws ArgumentError BulkDrag(coefficient=1e-3, time_discretization=IMEXFluxTimeDiscretization(1.0))

        drag = BulkDrag(coefficient=1e-3; time_discretization=imex)
        @test drag isa BulkDragBoundaryCondition
        @test occursin("IMEXFluxBoundaryCondition", sprint(show, drag))

        for arch in archs
            # A single cell with u = U₀ and no explicit part: a forward Euler step (the first AB2 step) followed
            # by the implicit step gives u¹ = U₀ / (1 + Cᴰ |U| Δt) exactly.
            grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
            kw = (advection=nothing, closure=nothing, timestepper=:QuasiAdamsBashforth2)
            Cᴰ = 0.1
            U₀ = 0.5
            Δt = 10.0 # Cᴰ |U| Δt / Δz = 0.5 for the quadratic and 1 for the linear drag

            for (formulation, λ) in ((QuadraticFormulation(), Cᴰ * U₀), (LinearFormulation(), Cᴰ))
                for side in (:bottom, :top)
                    u_bcs = FieldBoundaryConditions(; side => BulkDrag(formulation; coefficient=Cᴰ, time_discretization=imex))
                    model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs,), kw...)
                    bc = getproperty(model.velocities.u.boundary_conditions, side)
                    @test bc isa BulkDragBoundaryCondition
                    @test bc.classification.time_discretization isa IMEXFluxTimeDiscretization
                    @test bc.condition.direction isa XDirection

                    set!(model, u=U₀)
                    time_step!(model, Δt)
                    u¹ = Array(interior(model.velocities.u))[1, 1, 1]
                    @test u¹ ≈ U₀ / (1 + λ * Δt)
                end
            end

            # With a background velocity the explicit part λ U∞ contributes to the tendency, and the
            # implicit step then divides by 1 + Cᴰ |U| Δt with |U| evaluated after the explicit step.
            U∞ = 0.2
            u_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, background_velocities=(U∞, 0, 0), time_discretization=imex))
            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs,), kw...)
            set!(model, u=U₀)
            time_step!(model, Δt)
            u★ = U₀ - Δt * Cᴰ * (U₀ + U∞) * U∞
            u¹ = u★ / (1 + Cᴰ * (u★ + U∞) * Δt)
            @test Array(interior(model.velocities.u))[1, 1, 1] ≈ u¹

            # IMEX and explicit drag agree for a small time step
            Δt = 1e-4
            drags = (explicit = BulkDrag(coefficient=Cᴰ, background_velocities=(U∞, 0, 0)),
                     imex = BulkDrag(coefficient=Cᴰ, background_velocities=(U∞, 0, 0), time_discretization=imex))
            u¹ = map(drags) do drag
                model = NonhydrostaticModel(grid; boundary_conditions=(u=FieldBoundaryConditions(bottom=drag),), advection=nothing, closure=nothing)
                set!(model, u=U₀)
                time_step!(model, Δt)
                Array(interior(model.velocities.u))[1, 1, 1]
            end
            @test u¹.imex ≈ u¹.explicit rtol=1e-6
            @test u¹.imex != U₀

            # The explicit treatment is unstable for Cᴰ Δt / Δz > 2; the implicit one is not. Without a closure
            # only the bottom cell feels the drag.
            column = RectilinearGrid(arch, size=4, z=(-4, 0), topology=(Flat, Flat, Bounded))
            Cᴰ = 0.05
            Δt = 100.0 # Cᴰ Δt / Δz = 5

            u_bottom = map((explicit = BulkDrag(LinearFormulation(), coefficient=Cᴰ),
                        imex = BulkDrag(LinearFormulation(), coefficient=Cᴰ, time_discretization=imex))) do drag
                model = HydrostaticFreeSurfaceModel(column; boundary_conditions=(u=FieldBoundaryConditions(bottom=drag),),
                                                    momentum_advection=nothing, tracer_advection=nothing, tracers=(),
                                                    buoyancy=nothing, coriolis=nothing, closure=nothing)
                set!(model, u=1)
                for _ in 1:20
                    time_step!(model, Δt)
                end
                Array(interior(model.velocities.u))[1, 1, 1]
            end
            @test !(abs(u_bottom.explicit) < 1)
            @test 0 < u_bottom.imex < 1

            # Immersed bottom facet: the drag acts on the cell above the immersed bottom
            underlying = RectilinearGrid(arch, size=(1, 1, 4), x=(0, 1), y=(0, 1), z=(-4, 0))
            grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(-2))
            Cᴰ = 0.1
            Δt = 10.0
            drag = BulkDrag(LinearFormulation(), coefficient=Cᴰ, time_discretization=imex)
            u_bcs = FieldBoundaryConditions(immersed=ImmersedBoundaryCondition(bottom=drag))
            model = NonhydrostaticModel(grid; boundary_conditions=(u=u_bcs,), kw...)
            @test model.velocities.u.boundary_conditions.immersed.bottom isa BulkDragBoundaryCondition
            set!(model, u=U₀)
            time_step!(model, Δt)
            u = Array(interior(model.velocities.u))[1, 1, :]
            @test u[3] ≈ U₀ / (1 + Cᴰ * Δt)
            @test u[4] ≈ U₀

            # The same drag in a hydrostatic model with a split-explicit free surface decelerates the column
            model = HydrostaticFreeSurfaceModel(grid; boundary_conditions=(u=u_bcs,),
                                                momentum_advection=nothing, tracer_advection=nothing, tracers=(),
                                                buoyancy=nothing, coriolis=nothing, closure=nothing)
            set!(model, u=U₀)
            for _ in 1:10
                time_step!(model, Δt)
            end
            u = Array(interior(model.velocities.u))[1, 1, :]
            @test 0 < u[3] < U₀
            @test u[4] ≈ U₀

            # Only vertical boundaries and facets are supported
            xgrid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1), topology=(Bounded, Periodic, Bounded))
            @test_throws ErrorException NonhydrostaticModel(xgrid; boundary_conditions=(v=FieldBoundaryConditions(east=drag),), kw...)
            @test_throws ErrorException NonhydrostaticModel(grid; boundary_conditions=(v=FieldBoundaryConditions(immersed=ImmersedBoundaryCondition(east=drag)),), kw...)
        end
    end

    #####
    ##### Test similarity theory drag coefficient
    #####

    @testset "Similarity theory drag coefficient" begin
        ϰ = 0.4  # von Karman constant
        ℓ = 1e-4 # roughness length
        d₀ = 0.05 # distance to wall

        expected_Cᴰ = (ϰ / log(d₀ / ℓ))^2

        u_drag = BulkDrag(direction=XDirection(), coefficient=expected_Cᴰ)

        @test u_drag.condition.coefficient ≈ expected_Cᴰ
    end
end
