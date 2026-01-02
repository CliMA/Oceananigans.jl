include("dependencies_for_runtests.jl")

using Oceananigans.Grids: XDirection, YDirection, ZDirection
using Oceananigans.Models: BulkDrag, BulkDragFunction, BulkDragBoundaryCondition,
                           QuadraticFormulation, LinearFormulation
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
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
