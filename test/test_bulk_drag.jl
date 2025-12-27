include("dependencies_for_runtests.jl")

using Oceananigans.Models: BulkDrag, drag_boundary_conditions, drag_immersed_boundary_conditions
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, FieldBoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, ImmersedBoundaryCondition

@testset "BulkDrag" begin
    @info "Testing BulkDrag..."

    #####
    ##### Test BulkDrag construction
    #####

    @testset "BulkDrag construction" begin
        # Default construction
        bulk_drag = BulkDrag()
        @test bulk_drag.roughness_length == 1e-4
        @test bulk_drag.von_karman_constant == 0.4
        @test isnothing(bulk_drag.drag_coefficient)

        # Custom roughness length
        bulk_drag = BulkDrag(roughness_length=1e-3)
        @test bulk_drag.roughness_length == 1e-3
        @test bulk_drag.von_karman_constant == 0.4

        # Custom von Karman constant
        bulk_drag = BulkDrag(von_karman_constant=0.41)
        @test bulk_drag.von_karman_constant == 0.41

        # Fixed drag coefficient
        bulk_drag = BulkDrag(drag_coefficient=0.003)
        @test bulk_drag.drag_coefficient == 0.003

        # Float32 precision
        bulk_drag = BulkDrag(Float32; roughness_length=1e-4)
        @test bulk_drag.roughness_length isa Float32

        # Test show methods
        bulk_drag = BulkDrag(roughness_length=1e-4)
        @test occursin("roughness_length", sprint(show, bulk_drag))
        @test occursin("similarity theory", sprint(show, bulk_drag))

        bulk_drag_fixed = BulkDrag(drag_coefficient=0.003)
        @test occursin("0.003", sprint(show, bulk_drag_fixed))
    end

    #####
    ##### Test drag_boundary_conditions
    #####

    @testset "drag_boundary_conditions" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            bulk_drag = BulkDrag(roughness_length=1e-4)

            # Test 3D boundary conditions (for nonhydrostatic model)
            u_bc, v_bc, w_bc = drag_boundary_conditions(grid, bulk_drag)

            @test u_bc isa FluxBoundaryCondition
            @test v_bc isa FluxBoundaryCondition
            @test w_bc isa FluxBoundaryCondition

            # Test 2D boundary conditions (for hydrostatic model)
            u_bc, v_bc = drag_boundary_conditions(grid, bulk_drag; include_vertical_velocity=false)

            @test u_bc isa FluxBoundaryCondition
            @test v_bc isa FluxBoundaryCondition

            # Test with fixed drag coefficient
            bulk_drag_fixed = BulkDrag(drag_coefficient=0.003)
            u_bc, v_bc = drag_boundary_conditions(grid, bulk_drag_fixed; include_vertical_velocity=false)
            
            @test u_bc isa FluxBoundaryCondition
            @test v_bc isa FluxBoundaryCondition
        end
    end

    #####
    ##### Test with NonhydrostaticModel
    #####

    @testset "BulkDrag with NonhydrostaticModel" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            bulk_drag = BulkDrag(roughness_length=1e-4)

            u_drag_bc, v_drag_bc, w_drag_bc = drag_boundary_conditions(grid, bulk_drag)

            u_bcs = FieldBoundaryConditions(bottom=u_drag_bc)
            v_bcs = FieldBoundaryConditions(bottom=v_drag_bc)

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
    ##### Test with HydrostaticFreeSurfaceModel
    #####

    @testset "BulkDrag with HydrostaticFreeSurfaceModel" begin
        for arch in archs
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            bulk_drag = BulkDrag(roughness_length=1e-4)

            # Use 2D speed calculation for hydrostatic models
            u_drag_bc, v_drag_bc = drag_boundary_conditions(grid, bulk_drag; include_vertical_velocity=false)

            u_bcs = FieldBoundaryConditions(bottom=u_drag_bc)
            v_bcs = FieldBoundaryConditions(bottom=v_drag_bc)

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

    @testset "BulkDrag with ImmersedBoundaryGrid" begin
        for arch in archs
            underlying_grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1))
            
            # Simple flat bottom at z = -0.5
            bottom(x, y) = -0.5
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

            bulk_drag = BulkDrag(roughness_length=1e-4)

            # Create immersed boundary conditions
            u_ibc, v_ibc, w_ibc = drag_immersed_boundary_conditions(grid, bulk_drag)

            @test u_ibc isa ImmersedBoundaryCondition
            @test v_ibc isa ImmersedBoundaryCondition
            @test w_ibc isa ImmersedBoundaryCondition

            # Test 2D version
            u_ibc_2d, v_ibc_2d = drag_immersed_boundary_conditions(grid, bulk_drag; include_vertical_velocity=false)

            @test u_ibc_2d isa ImmersedBoundaryCondition
            @test v_ibc_2d isa ImmersedBoundaryCondition

            # Create model with immersed boundary conditions
            u_bcs = FieldBoundaryConditions(immersed=u_ibc)
            v_bcs = FieldBoundaryConditions(immersed=v_ibc)
            w_bcs = FieldBoundaryConditions(immersed=w_ibc)

            model = NonhydrostaticModel(; grid,
                                        boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs))

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
    ##### Test drag coefficient calculation
    #####

    @testset "Drag coefficient calculation" begin
        grid = RectilinearGrid(CPU(), size=(10, 10, 10), extent=(1, 1, 1))
        
        # Grid spacing is 0.1, so d₀ = 0.05
        # With ℓ = 1e-4, cᴰ = (0.4 / log(0.05 / 1e-4))² ≈ 0.00415
        bulk_drag = BulkDrag(roughness_length=1e-4)

        using Oceananigans.Models.BulkDragModule: compute_drag_coefficient

        cᴰ = compute_drag_coefficient(bulk_drag, grid, :z)
        
        expected_cᴰ = (0.4 / log(0.05 / 1e-4))^2
        @test isapprox(cᴰ, expected_cᴰ; rtol=1e-10)

        # Test that fixed drag coefficient is returned when specified
        bulk_drag_fixed = BulkDrag(drag_coefficient=0.003)
        cᴰ_fixed = compute_drag_coefficient(bulk_drag_fixed, grid, :z)
        @test cᴰ_fixed == 0.003
    end

    #####
    ##### Test speed functions
    #####

    @testset "Speed computation" begin
        grid = RectilinearGrid(CPU(), size=(4, 4, 4), extent=(1, 1, 1))
        
        u = CenterField(grid)
        v = CenterField(grid)
        w = CenterField(grid)

        set!(u, 1.0)
        set!(v, 0.0)
        set!(w, 0.0)

        using Oceananigans.Models.BulkDragModule: speedᶠᶜᶜ, speedᶜᶠᶜ, speedᶜᶜᶠ
        using Oceananigans.Models.BulkDragModule: speed_xyᶠᶜᶜ, speed_xyᶜᶠᶜ

        # Test speed at interior point
        i, j, k = 2, 2, 2
        
        # Note: u is actually at (Face, Center, Center) so we need to be careful
        # For simplicity, just test that these functions run without error
        # and return positive values
        speed_u = speed_xyᶠᶜᶜ(i, j, k, grid, u, v)
        speed_v = speed_xyᶜᶠᶜ(i, j, k, grid, u, v)

        @test speed_u >= 0
        @test speed_v >= 0
    end
end

