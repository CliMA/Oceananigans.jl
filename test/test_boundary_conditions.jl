using Oceananigans.BoundaryConditions: PBC, NFBC, NPBC

function instantiate_boundary_function(B, X1, X2, func)
    boundary_function = BoundaryFunction{B, X1, X2}(func)
    return true
end

function instantiate_tracer_boundary_condition(bctype, B, func)
    boundary_condition = TracerBoundaryCondition(bctype, B, func)
    return true
end

function instantiate_u_boundary_condition(bctype, B, func)
    boundary_condition = UVelocityBoundaryCondition(bctype, B, func)
    return true
end

function instantiate_v_boundary_condition(bctype, B, func)
    boundary_condition = VVelocityBoundaryCondition(bctype, B, func)
    return true
end

function instantiate_w_boundary_condition(bctype, B, func)
    boundary_condition = WVelocityBoundaryCondition(bctype, B, func)
    return true
end

@testset "Boundary conditions" begin
    @info "Testing boundary conditions..."

    @testset "Boundary functions" begin
        @info "  Testing boundary functions..."

        simple_bc(ξ, η, t) = exp(ξ) * cos(η) * sin(t)
        for B in (:x, :y, :z)
            for X1 in (:Face, :Cell)
                @test instantiate_boundary_function(B, X1, Cell, simple_bc)
            end

            for bctype in (Value, Gradient)
                @test instantiate_tracer_boundary_condition(bctype, B, simple_bc)
                @test instantiate_u_boundary_condition(bctype, B, simple_bc)
                @test instantiate_v_boundary_condition(bctype, B, simple_bc)
                @test instantiate_w_boundary_condition(bctype, B, simple_bc)
            end
        end
    end

    @testset "Field boundary conditions" begin
        @info "  Testing field boundary functions..."

        ppb_topology = (Periodic, Periodic, Bounded)
        ppb_grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=ppb_topology)

        u_bcs = UVelocityBoundaryConditions(ppb_grid)
        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa PBC
        @test u_bcs.x.right isa PBC
        @test u_bcs.y.left  isa PBC
        @test u_bcs.y.right isa PBC
        @test u_bcs.z.left  isa NFBC
        @test u_bcs.z.right isa NFBC

        v_bcs = VVelocityBoundaryConditions(ppb_grid)
        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa PBC
        @test v_bcs.x.right isa PBC
        @test v_bcs.y.left  isa PBC
        @test v_bcs.y.right isa PBC
        @test v_bcs.z.left  isa NFBC
        @test v_bcs.z.right isa NFBC

        w_bcs = WVelocityBoundaryConditions(ppb_grid)
        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa PBC
        @test w_bcs.x.right isa PBC
        @test w_bcs.y.left  isa PBC
        @test w_bcs.y.right isa PBC
        @test w_bcs.z.left  isa NPBC
        @test w_bcs.z.right isa NPBC

        Tbcs = TracerBoundaryConditions(ppb_grid)
        @test Tbcs isa FieldBoundaryConditions
        @test Tbcs.x.left  isa PBC
        @test Tbcs.x.right isa PBC
        @test Tbcs.y.left  isa PBC
        @test Tbcs.y.right isa PBC
        @test Tbcs.z.left  isa NFBC
        @test Tbcs.z.right isa NFBC

        pbb_topology = (Periodic, Bounded, Bounded)
        pbb_grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=pbb_topology)

        u_bcs = UVelocityBoundaryConditions(pbb_grid)
        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa PBC
        @test u_bcs.x.right isa PBC
        @test u_bcs.y.left  isa NFBC
        @test u_bcs.y.right isa NFBC
        @test u_bcs.z.left  isa NFBC
        @test u_bcs.z.right isa NFBC

        v_bcs = VVelocityBoundaryConditions(pbb_grid)
        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa PBC
        @test v_bcs.x.right isa PBC
        @test v_bcs.y.left  isa NPBC
        @test v_bcs.y.right isa NPBC
        @test v_bcs.z.left  isa NFBC
        @test v_bcs.z.right isa NFBC

        w_bcs = WVelocityBoundaryConditions(pbb_grid)
        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa PBC
        @test w_bcs.x.right isa PBC
        @test w_bcs.y.left  isa NFBC
        @test w_bcs.y.right isa NFBC
        @test w_bcs.z.left  isa NPBC
        @test w_bcs.z.right isa NPBC

        Tbcs = TracerBoundaryConditions(pbb_grid)
        @test Tbcs isa FieldBoundaryConditions
        @test Tbcs.x.left  isa PBC
        @test Tbcs.x.right isa PBC
        @test Tbcs.y.left  isa NFBC
        @test Tbcs.y.right isa NFBC
        @test Tbcs.z.left  isa NFBC
        @test Tbcs.z.right isa NFBC
    end
end
