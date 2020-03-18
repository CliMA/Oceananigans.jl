using Oceananigans.BoundaryConditions: PBC, ZFBC, NFBC

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

        # Triply periodic
        ppp_topology = (Periodic, Periodic, Periodic)
        ppp_grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=ppp_topology)

        u_bcs = UVelocityBoundaryConditions(ppp_grid)
        v_bcs = VVelocityBoundaryConditions(ppp_grid)
        w_bcs = WVelocityBoundaryConditions(ppp_grid)
        T_bcs = TracerBoundaryConditions(ppp_grid)

        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa PBC
        @test u_bcs.x.right isa PBC
        @test u_bcs.y.left  isa PBC
        @test u_bcs.y.right isa PBC
        @test u_bcs.z.left  isa PBC
        @test u_bcs.z.right isa PBC

        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa PBC
        @test v_bcs.x.right isa PBC
        @test v_bcs.y.left  isa PBC
        @test v_bcs.y.right isa PBC
        @test v_bcs.z.left  isa PBC
        @test v_bcs.z.right isa PBC

        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa PBC
        @test w_bcs.x.right isa PBC
        @test w_bcs.y.left  isa PBC
        @test w_bcs.y.right isa PBC
        @test w_bcs.z.left  isa PBC
        @test w_bcs.z.right isa PBC

        @test T_bcs isa FieldBoundaryConditions
        @test T_bcs.x.left  isa PBC
        @test T_bcs.x.right isa PBC
        @test T_bcs.y.left  isa PBC
        @test T_bcs.y.right isa PBC
        @test T_bcs.z.left  isa PBC
        @test T_bcs.z.right isa PBC

        # Doubly periodic. Engineers call this a "Channel geometry".
        ppb_topology = (Periodic, Periodic, Bounded)
        ppb_grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=ppb_topology)

        u_bcs = UVelocityBoundaryConditions(ppb_grid)
        v_bcs = VVelocityBoundaryConditions(ppb_grid)
        w_bcs = WVelocityBoundaryConditions(ppb_grid)
        T_bcs = TracerBoundaryConditions(ppb_grid)

        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa PBC
        @test u_bcs.x.right isa PBC
        @test u_bcs.y.left  isa PBC
        @test u_bcs.y.right isa PBC
        @test u_bcs.z.left  isa ZFBC
        @test u_bcs.z.right isa ZFBC

        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa PBC
        @test v_bcs.x.right isa PBC
        @test v_bcs.y.left  isa PBC
        @test v_bcs.y.right isa PBC
        @test v_bcs.z.left  isa ZFBC
        @test v_bcs.z.right isa ZFBC

        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa PBC
        @test w_bcs.x.right isa PBC
        @test w_bcs.y.left  isa PBC
        @test w_bcs.y.right isa PBC
        @test w_bcs.z.left  isa NFBC
        @test w_bcs.z.right isa NFBC

        @test T_bcs isa FieldBoundaryConditions
        @test T_bcs.x.left  isa PBC
        @test T_bcs.x.right isa PBC
        @test T_bcs.y.left  isa PBC
        @test T_bcs.y.right isa PBC
        @test T_bcs.z.left  isa ZFBC
        @test T_bcs.z.right isa ZFBC

        # Singly periodic. Oceanographers call this a "Channel", engineers call it a "Pipe"
        pbb_topology = (Periodic, Bounded, Bounded)
        pbb_grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=pbb_topology)

        u_bcs = UVelocityBoundaryConditions(pbb_grid)
        v_bcs = VVelocityBoundaryConditions(pbb_grid)
        w_bcs = WVelocityBoundaryConditions(pbb_grid)
        T_bcs = TracerBoundaryConditions(pbb_grid)

        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa PBC
        @test u_bcs.x.right isa PBC
        @test u_bcs.y.left  isa ZFBC
        @test u_bcs.y.right isa ZFBC
        @test u_bcs.z.left  isa ZFBC
        @test u_bcs.z.right isa ZFBC

        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa PBC
        @test v_bcs.x.right isa PBC
        @test v_bcs.y.left  isa NFBC
        @test v_bcs.y.right isa NFBC
        @test v_bcs.z.left  isa ZFBC
        @test v_bcs.z.right isa ZFBC

        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa PBC
        @test w_bcs.x.right isa PBC
        @test w_bcs.y.left  isa ZFBC
        @test w_bcs.y.right isa ZFBC
        @test w_bcs.z.left  isa NFBC
        @test w_bcs.z.right isa NFBC

        @test T_bcs isa FieldBoundaryConditions
        @test T_bcs.x.left  isa PBC
        @test T_bcs.x.right isa PBC
        @test T_bcs.y.left  isa ZFBC
        @test T_bcs.y.right isa ZFBC
        @test T_bcs.z.left  isa ZFBC
        @test T_bcs.z.right isa ZFBC

        # Triply bounded. Oceanographers call this a "Basin", engineers call it a "Box"
        bbb_topology = (Bounded, Bounded, Bounded)
        bbb_grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), topology=bbb_topology)

        u_bcs = UVelocityBoundaryConditions(bbb_grid)
        v_bcs = VVelocityBoundaryConditions(bbb_grid)
        w_bcs = WVelocityBoundaryConditions(bbb_grid)
        T_bcs = TracerBoundaryConditions(bbb_grid)

        @test u_bcs isa FieldBoundaryConditions
        @test u_bcs.x.left  isa NFBC
        @test u_bcs.x.right isa NFBC
        @test u_bcs.y.left  isa ZFBC
        @test u_bcs.y.right isa ZFBC
        @test u_bcs.z.left  isa ZFBC
        @test u_bcs.z.right isa ZFBC

        @test v_bcs isa FieldBoundaryConditions
        @test v_bcs.x.left  isa ZFBC
        @test v_bcs.x.right isa ZFBC
        @test v_bcs.y.left  isa NFBC
        @test v_bcs.y.right isa NFBC
        @test v_bcs.z.left  isa ZFBC
        @test v_bcs.z.right isa ZFBC

        @test w_bcs isa FieldBoundaryConditions
        @test w_bcs.x.left  isa ZFBC
        @test w_bcs.x.right isa ZFBC
        @test w_bcs.y.left  isa ZFBC
        @test w_bcs.y.right isa ZFBC
        @test w_bcs.z.left  isa NFBC
        @test w_bcs.z.right isa NFBC

        @test T_bcs isa FieldBoundaryConditions
        @test T_bcs.x.left  isa ZFBC
        @test T_bcs.x.right isa ZFBC
        @test T_bcs.y.left  isa ZFBC
        @test T_bcs.y.right isa ZFBC
        @test T_bcs.z.left  isa ZFBC
        @test T_bcs.z.right isa ZFBC
    end
end
