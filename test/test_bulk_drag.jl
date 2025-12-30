include("dependencies_for_runtests.jl")

using Oceananigans.Grids: XDirection, YDirection
using Oceananigans.Models: BulkDrag, BulkDragFunction, BulkDragBoundaryCondition
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, ImmersedBoundaryCondition

@testset "BulkDrag" begin
    @info "Testing BulkDrag..."

    #####
    ##### Test BulkDragFunction construction
    #####

    @testset "BulkDragFunction construction" begin
        # Default construction
        df = BulkDragFunction()
        @test isnothing(df.direction)
        @test df.coefficient == 1e-3
        @test df.background_velocities == (0, 0)

        # With direction
        df_x = BulkDragFunction(direction=XDirection())
        @test df_x.direction isa XDirection
        @test df_x.coefficient == 1e-3

        df_y = BulkDragFunction(direction=YDirection())
        @test df_y.direction isa YDirection

        # Custom coefficient
        df = BulkDragFunction(coefficient=0.003)
        @test df.coefficient == 0.003

        # Custom background velocities
        df = BulkDragFunction(background_velocities=(0.1, 0.2))
        @test df.background_velocities == (0.1, 0.2)

        # Test show method
        @test occursin("BulkDragFunction", sprint(show, df))
    end

    #####
    ##### Test BulkBottomDrag boundary condition
    #####

    @testset "BulkBottomDrag boundary condition" begin
        u_drag = BulkBottomDrag(direction=XDirection(), coefficient=1e-3)
        v_drag = BulkBottomDrag(direction=YDirection(), coefficient=1e-3)

        @test u_drag isa BulkDragBoundaryCondition
        @test v_drag isa BulkDragBoundaryCondition

        @test u_drag.condition.direction isa XDirection
        @test v_drag.condition.direction isa YDirection
    end

    #####
    ##### Test automatic direction inference with NonhydrostaticModel
    #####

    @testset "Automatic direction inference with NonhydrostaticModel" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            
            # Use the same BulkBottomDrag for both u and v - direction is inferred automatically
            Cᴰ = 0.003
            drag_bc = BulkBottomDrag(coefficient=Cᴰ)

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            v_bcs = FieldBoundaryConditions(bottom=drag_bc)

            model = NonhydrostaticModel(; grid, 
                                        boundary_conditions=(u=u_bcs, v=v_bcs))

            # Check that direction was inferred correctly
            @test model.velocities.u.boundary_conditions.bottom.condition.direction isa XDirection
            @test model.velocities.v.boundary_conditions.bottom.condition.direction isa YDirection

            # Set initial velocity field
            set!(model, u=0.1, v=0.1)

            # Time step the model
            simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
            run!(simulation)

            # Check that the simulation ran without errors
            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test with background velocities
    #####

    @testset "BulkBottomDrag with background velocities" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            
            Cᴰ = 0.003
            V∞ = 0.1  # Background along-y velocity

            drag_bc = BulkBottomDrag(coefficient=Cᴰ, background_velocities=(0, V∞))

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            v_bcs = FieldBoundaryConditions(bottom=drag_bc)

            model = NonhydrostaticModel(; grid, 
                                        boundary_conditions=(u=u_bcs, v=v_bcs))

            # Check background velocities are stored correctly
            @test model.velocities.u.boundary_conditions.bottom.condition.background_velocities == (0, V∞)
            @test model.velocities.v.boundary_conditions.bottom.condition.background_velocities == (0, V∞)

            # Set initial velocity field
            set!(model, u=0.1, v=0.1)

            # Time step the model
            simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
            run!(simulation)

            # Check that the simulation ran without errors
            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test with HydrostaticFreeSurfaceModel
    #####

    @testset "BulkBottomDrag with HydrostaticFreeSurfaceModel" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            
            # Use automatic direction inference
            Cᴰ = 0.003
            drag_bc = BulkBottomDrag(coefficient=Cᴰ)

            u_bcs = FieldBoundaryConditions(bottom=drag_bc)
            v_bcs = FieldBoundaryConditions(bottom=drag_bc)

            model = HydrostaticFreeSurfaceModel(; grid, 
                                                 boundary_conditions=(u=u_bcs, v=v_bcs))

            # Set initial velocity field
            set!(model, u=0.1, v=0.1)

            # Time step the model
            simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
            run!(simulation)

            # Check that the simulation ran without errors
            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test with ImmersedBoundaryGrid
    #####

    @testset "BulkBottomDrag with ImmersedBoundaryGrid" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            
            # Create a simple bottom topography
            bottom(x, y) = -0.5
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
            
            Cᴰ = 0.003
            drag_bc = BulkBottomDrag(coefficient=Cᴰ)

            # Apply to domain bottom and only the bottom facet of immersed boundaries
            u_bcs = FieldBoundaryConditions(bottom=drag_bc, immersed=ImmersedBoundaryCondition(bottom=drag_bc))
            v_bcs = FieldBoundaryConditions(bottom=drag_bc, immersed=ImmersedBoundaryCondition(bottom=drag_bc))

            model = NonhydrostaticModel(; grid, 
                                        boundary_conditions=(u=u_bcs, v=v_bcs))

            # Set initial velocity field
            set!(model, u=0.1, v=0.1)

            # Time step the model
            simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
            run!(simulation)

            # Check that the simulation ran without errors
            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test with ImmersedBoundaryCondition
    #####

    @testset "BulkBottomDrag with ImmersedBoundaryCondition" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            
            # Create a simple bottom topography
            bottom(x, y) = -0.5
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
            
            Cᴰ = 0.003
            drag_bc = BulkBottomDrag(coefficient=Cᴰ)
            
            # Use ImmersedBoundaryCondition to specify bottom drag only
            u_ibc = ImmersedBoundaryCondition(bottom=drag_bc)
            v_ibc = ImmersedBoundaryCondition(bottom=drag_bc)

            u_bcs = FieldBoundaryConditions(immersed=u_ibc)
            v_bcs = FieldBoundaryConditions(immersed=v_ibc)

            model = NonhydrostaticModel(; grid, 
                                        boundary_conditions=(u=u_bcs, v=v_bcs))

            # Set initial velocity field
            set!(model, u=0.1, v=0.1)

            # Time step the model
            simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
            run!(simulation)

            # Check that the simulation ran without errors
            @test model.clock.iteration == 3
        end
    end

    #####
    ##### Test similarity theory drag coefficient
    #####

    @testset "Similarity theory drag coefficient" begin
        # Test computing drag coefficient from similarity theory
        # cᴰ = (ϰ / log(d₀ / ℓ))²
        
        ϰ = 0.4  # von Karman constant
        ℓ = 1e-4 # roughness length
        d₀ = 0.05 # distance to wall (half grid spacing for Δz = 0.1)
        
        expected_Cᴰ = (ϰ / log(d₀ / ℓ))^2
        
        # Create drag with this coefficient
        u_drag = BulkBottomDrag(direction=XDirection(), coefficient=expected_Cᴰ)
        
        @test u_drag.condition.coefficient ≈ expected_Cᴰ
    end
end
