using Oceananigans.BoundaryConditions: PBC, ZFBC, NFBC, BoundaryFunction, ParameterizedDiscreteBoundaryFunction

using Oceananigans.Fields: Face, Cell

function instantiate_boundary_function(B, X1, X2, func)
    boundary_function = BoundaryFunction{B, X1, X2}(func)
    return true
end

bc_location(bc::BoundaryFunction{B, X, Y}) where {B, X, Y} = (B, X, Y)

simple_bc(ξ, η, t) = exp(ξ) * cos(η) * sin(t)
simple_parameterized_bc(ξ, η, t, p) = p.a * exp(ξ) * cos(η) * sin(t)

complicated_bc(i, j, grid, clock, state) = rand()
complicated_parameterized_bc(i, j, grid, clock, state, p) = p.a * rand()

@testset "Boundary conditions" begin
    @info "Testing boundary conditions..."

    @testset "Boundary functions" begin
        @info "  Testing boundary functions..."

        for B in (:x, :y, :z)
            for X1 in (:Face, :Cell)
                @test instantiate_boundary_function(B, X1, Cell, simple_bc)
            end
        end

        bc = BoundaryCondition(Value, simple_bc)
        @test typeof(bc.condition) <: BoundaryFunction
        @test bc.condition.func === simple_bc

        bc = BoundaryCondition(Value, complicated_bc; discrete_form=true)
        @test bc.condition === complicated_bc

        bc = BoundaryCondition(Value, simple_parameterized_bc; parameters=(a=π,))
        @test bc.condition.parameters.a == π

        bc = BoundaryCondition(Value, complicated_parameterized_bc; parameters=(a=π,), discrete_form=true)
        @test typeof(bc.condition) <: ParameterizedDiscreteBoundaryFunction
        @test bc.condition.func === complicated_parameterized_bc
    end

    @testset "Field and coordinate boundary conditions" begin
        @info "  Testing field and coordinate boundary conditions..."

        # Triply periodic
        ppp_topology = (Periodic, Periodic, Periodic)
        ppp_grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=ppp_topology)

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
        ppb_grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=ppb_topology)

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
        pbb_grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=pbb_topology)

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
        bbb_grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1), topology=bbb_topology)

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

        grid = bbb_grid

        u_bcs = UVelocityBoundaryConditions(grid;
                                              east = BoundaryCondition(NormalFlow, simple_bc), 
                                              west = BoundaryCondition(NormalFlow, simple_bc),
                                            bottom = BoundaryCondition(Value, simple_bc), 
                                               top = BoundaryCondition(Value, simple_bc), 
                                             north = BoundaryCondition(Value, simple_bc), 
                                             south = BoundaryCondition(Value, simple_bc)
                                           )

        @test bc_location(u_bcs.east.condition) === (:x, Cell, Cell)
        @test bc_location(u_bcs.west.condition) === (:x, Cell, Cell)
        @test bc_location(u_bcs.north.condition) === (:y, Face, Cell)
        @test bc_location(u_bcs.south.condition) === (:y, Face, Cell)
        @test bc_location(u_bcs.top.condition) === (:z, Face, Cell)
        @test bc_location(u_bcs.bottom.condition) === (:z, Face, Cell)

        v_bcs = VVelocityBoundaryConditions(grid;
                                              east = BoundaryCondition(Value, simple_bc), 
                                              west = BoundaryCondition(Value, simple_bc),
                                            bottom = BoundaryCondition(NormalFlow, simple_bc), 
                                               top = BoundaryCondition(NormalFlow, simple_bc), 
                                             north = BoundaryCondition(Value, simple_bc), 
                                             south = BoundaryCondition(Value, simple_bc)
                                           )

        @test bc_location(v_bcs.east.condition) === (:x, Face, Cell)
        @test bc_location(v_bcs.west.condition) === (:x, Face, Cell)
        @test bc_location(v_bcs.north.condition) === (:y, Cell, Cell)
        @test bc_location(v_bcs.south.condition) === (:y, Cell, Cell)
        @test bc_location(v_bcs.top.condition) === (:z, Cell, Face)
        @test bc_location(v_bcs.bottom.condition) === (:z, Cell, Face)

        w_bcs = WVelocityBoundaryConditions(grid;
                                              east = BoundaryCondition(Value, simple_bc), 
                                              west = BoundaryCondition(Value, simple_bc),
                                            bottom = BoundaryCondition(Value, simple_bc), 
                                               top = BoundaryCondition(Value, simple_bc), 
                                             north = BoundaryCondition(NormalFlow, simple_bc), 
                                             south = BoundaryCondition(NormalFlow, simple_bc)
                                           )

        @test bc_location(w_bcs.east.condition) === (:x, Cell, Face)
        @test bc_location(w_bcs.west.condition) === (:x, Cell, Face)
        @test bc_location(w_bcs.north.condition) === (:y, Cell, Face)
        @test bc_location(w_bcs.south.condition) === (:y, Cell, Face)
        @test bc_location(w_bcs.top.condition) === (:z, Cell, Cell)
        @test bc_location(w_bcs.bottom.condition) === (:z, Cell, Cell)

        T_bcs = TracerBoundaryConditions(grid;
                                           east = BoundaryCondition(Value, simple_bc),
                                           west = BoundaryCondition(Value, simple_bc),
                                         bottom = BoundaryCondition(Value, simple_bc),
                                            top = BoundaryCondition(Value, simple_bc),
                                          north = BoundaryCondition(Value, simple_bc),
                                          south = BoundaryCondition(Value, simple_bc)
                                         )

        @test bc_location(T_bcs.east.condition) === (:x, Cell, Cell)
        @test bc_location(T_bcs.west.condition) === (:x, Cell, Cell)
        @test bc_location(T_bcs.north.condition) === (:y, Cell, Cell)
        @test bc_location(T_bcs.south.condition) === (:y, Cell, Cell)
        @test bc_location(T_bcs.top.condition) === (:z, Cell, Cell)
        @test bc_location(T_bcs.bottom.condition) === (:z, Cell, Cell)

        # Test boundary condition getting and setting 
        # Note that boundary condition setting is valid only in cases where
        # boundary conditionsare same type
        
        one_bc = BoundaryCondition(Value, 1.0)
        two_bc = BoundaryCondition(Value, 2.0)

        T_bcs = TracerBoundaryConditions(grid;
                                           east = one_bc,
                                           west = one_bc,
                                         bottom = one_bc,
                                            top = one_bc,
                                          north = one_bc,
                                          south = one_bc,
                                         )

        @test T_bcs.east   === one_bc
        @test T_bcs.west   === one_bc
        @test T_bcs.north  === one_bc
        @test T_bcs.south  === one_bc
        @test T_bcs.top    === one_bc
        @test T_bcs.bottom === one_bc

        T_bcs.east   = two_bc
        T_bcs.west   = two_bc
        T_bcs.north  = two_bc
        T_bcs.south  = two_bc
        T_bcs.top    = two_bc
        T_bcs.bottom = two_bc

        @test T_bcs.east   === two_bc
        @test T_bcs.west   === two_bc
        @test T_bcs.north  === two_bc
        @test T_bcs.south  === two_bc
        @test T_bcs.top    === two_bc
        @test T_bcs.bottom === two_bc
    end
end
