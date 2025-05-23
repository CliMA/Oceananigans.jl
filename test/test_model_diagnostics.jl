include("dependencies_for_runtests.jl")

using Oceananigans.Models: BoundaryConditionOperation, BoundaryConditionField

u_bottom_drag_continuous(x, y, t, grid, u, v, Cᴰ) = -Cᴰ * u * sqrt(u^2 + v^2)
v_bottom_drag_discrete(i, j, grid, clock, fields, Cᴰ) = - @inbounds Cᴰ * fields.v[i, j, 1]

v_west_drag_discrete(j, k, grid, clock, fields, Cᴰ)  = - @inbounds Cᴰ * fields.v[1, j, k]
v_east_drag_discrete(j, k, grid, clock, fields, Cᴰ)  = + @inbounds Cᴰ * fields.v[grid.Nx, j, k]
u_south_drag_discrete(i, k, grid, clock, fields, Cᴰ) = - @inbounds Cᴰ * fields.u[i, 1, k]
u_north_drag_discrete(i, k, grid, clock, fields, Cᴰ) = + @inbounds Cᴰ * fields.u[i, grid.Ny, k]

c_top_flux(i, j, grid, clock, fields, t★) = + @inbounds fields.c[i, j, grid.Nz] / t★
c_bottom_flux(i, j, grid, clock, fields, t★) = - @inbounds fields.c[i, j, 1] / t★

@testset "BoundaryConditionOperation and BoundaryConditionField" begin
    for arch in archs
        grid = RectilinearGrid(arch,
                               size=(4, 4, 4),
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 1),
                               topology=(Bounded, Bounded, Bounded))

        τx = Field{Center, Center, Nothing}(grid)
        set!(τx, -1e-4)

        u_wind = FluxBoundaryCondition(τx)
        v_wind = FluxBoundaryCondition(-1e-4)
        u_bottom_drag = FluxBoundaryCondition(u_bottom_drag_continuous, field_dependencies=:u, parameters=1e-3)
        v_bottom_drag = FluxBoundaryCondition(v_bottom_drag_discrete, discrete_form=true, parameters=1e-3)
        u_south_drag = FluxBoundaryCondition(u_south_drag_discrete, discrete_form=true, parameters=1e-3)
        u_north_drag = FluxBoundaryCondition(u_north_drag_discrete, discrete_form=true, parameters=1e-3)
        v_east_drag = FluxBoundaryCondition(v_east_drag_discrete, discrete_form=true, parameters=1e-3)
        v_west_drag = FluxBoundaryCondition(v_west_drag_discrete, discrete_form=true, parameters=1e-3)

        u_bcs = FieldBoundaryConditions(top=u_wind, bottom=u_bottom_drag,
                                        south=u_south_drag, north=u_north_drag)

        v_bcs = FieldBoundaryConditions(top=v_wind, bottom=v_bottom_drag,
                                        east=v_east_drag, west=v_west_drag)

        t★ = π
        c_top_bc = FluxBoundaryCondition(c_top_flux, discrete_form=true, parameters=t★)
        c_bottom_bc = FluxBoundaryCondition(c_bottom_flux, discrete_form=true, parameters=t★)
        c_bcs = FieldBoundaryConditions(top=c_top_bc, bottom=c_bottom_bc)

        nonhydrostatic_model = NonhydrostaticModel(; grid, tracers=:c, boundary_conditions=(u=u_bcs, v=v_bcs, c=c_bcs))
        hydrostatic_model = HydrostaticFreeSurfaceModel(; grid, tracers=:c, boundary_conditions=(u=u_bcs, v=v_bcs, c=c_bcs))

        for model in (nonhydrostatic_model, hydrostatic_model)
            M = typeof(model)
            @testset "BoundaryConditionOperation and BoundaryConditionField with $M" begin
                u, v, w = model.velocities

                u_bottom_bc = BoundaryConditionOperation(u, :bottom, model)
                u_top_bc    = BoundaryConditionOperation(u, :top, model)
                u_south_bc  = BoundaryConditionOperation(u, :south, model)
                u_north_bc  = BoundaryConditionOperation(u, :north, model)

                v_bottom_bc = BoundaryConditionOperation(v, :bottom, model)
                v_top_bc    = BoundaryConditionOperation(v, :top, model)
                v_west_bc   = BoundaryConditionOperation(v, :west, model)
                v_east_bc   = BoundaryConditionOperation(v, :east, model)

                c = model.tracers.c
                c_bottom_bc = BoundaryConditionOperation(c, :bottom, model)
                c_top_bc = BoundaryConditionOperation(c, :top, model)

                for bc in (u_bottom_bc, u_top_bc, u_south_bc, u_north_bc,
                        v_bottom_bc, v_top_bc, v_west_bc, v_east_bc,
                        c_bottom_bc, c_top_bc)
                    @test bc isa BoundaryConditionOperation
                end

                @test location(u_bottom_bc) == (Face, Center, Nothing)
                @test location(u_top_bc)    == (Face, Center, Nothing)
                @test location(u_south_bc)  == (Face, Nothing, Center)
                @test location(u_north_bc)  == (Face, Nothing, Center)

                @test location(v_bottom_bc) == (Center, Face, Nothing)
                @test location(v_top_bc)    == (Center, Face, Nothing)
                @test location(v_west_bc)   == (Nothing, Face, Center)
                @test location(v_east_bc)   == (Nothing, Face, Center)

                @test location(c_bottom_bc) == (Center, Center, Nothing)
                @test location(c_top_bc)    == (Center, Center, Nothing)
            
                initial_c(x, y, z) = 1 + 1 * (z > 0)
                set!(model, c=initial_c)
            
                c_bottom_bc_field = BoundaryConditionField(c, :bottom, model)
                c_top_bc_field = BoundaryConditionField(c, :top, model)

                compute!(c_bottom_bc_field)
                compute!(c_top_bc_field)

                @test c_bottom_bc_field isa BoundaryConditionField
                @test c_top_bc_field isa BoundaryConditionField
                
                @test all(interior(c_bottom_bc_field) .≈ - 1 / t★)
                @test all(interior(c_top_bc_field) .≈ + 2 / t★)
            end
        end
    end
end

