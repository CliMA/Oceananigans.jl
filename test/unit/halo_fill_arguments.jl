include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.BoundaryConditions: halo_fill_args, needs_halo_fill_args,
    ImpenetrableBoundaryCondition, MixedBoundaryCondition

@testset "Arguments passed to halo-filling kernels" begin
    grid = RectilinearGrid(CPU(); size=(4, 5, 6), extent=(1, 2, 3), topology=(Periodic, Bounded, Bounded))
    clock = Clock(grid)
    c = CenterField(grid)
    args = (clock, (; c))

    argument_free_bcs = (nothing,
                         missing,
                         PeriodicBoundaryCondition(),
                         NoFluxBoundaryCondition(),
                         FluxBoundaryCondition(1),
                         FluxBoundaryCondition((x, y, t) -> t), # flux halos copy the interior, whatever the flux
                         ValueBoundaryCondition(1),
                         GradientBoundaryCondition(0.5),
                         ImpenetrableBoundaryCondition(),
                         NormalFlowBoundaryCondition(2),
                         ValueBoundaryCondition(zeros(4, 6)),
                         MixedBoundaryCondition(1, 2))

    discrete_time(i, j, grid, clock, fields) = clock.time

    argument_reading_bcs = (ValueBoundaryCondition((x, y, t) -> t),
                            GradientBoundaryCondition(discrete_time; discrete_form=true),
                            NormalFlowBoundaryCondition((y, z, t) -> t),
                            NormalFlowBoundaryCondition(1; scheme=PerturbationAdvection(inflow_timescale=1)),
                            ValueBoundaryCondition(Field{Center, Nothing, Center}(grid)),
                            MixedBoundaryCondition((x, y, t) -> t, 0))

    @testset "Argument-free boundary condition $n" for (n, bc) in enumerate(argument_free_bcs)
        @test !needs_halo_fill_args(bc)
        @test @inferred(halo_fill_args((bc,), args)) === ()
        @test @inferred(halo_fill_args((bc, bc), args)) === ()
    end

    @testset "Argument-reading boundary condition $n" for (n, bc) in enumerate(argument_reading_bcs)
        @test needs_halo_fill_args(bc)
        @test @inferred(halo_fill_args((bc,), args)) === args
        @test @inferred(halo_fill_args((bc, nothing), args)) === args
        @test @inferred(halo_fill_args((NoFluxBoundaryCondition(), bc), args)) === args
    end

    @test !needs_halo_fill_args(c.boundary_conditions)
    @test @inferred(halo_fill_args(c.boundary_conditions, args)) === ()
end

@testset "Halo filling with and without boundary-condition arguments [$(typeof(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size=(4, 5, 6), extent=(1, 2, 3), topology=(Periodic, Bounded, Bounded))
    clock = Clock(grid)
    clock.time = 3

    ccc = (Center(), Center(), Center())

    argument_free_bcs = FieldBoundaryConditions(grid, ccc;
                                                south = ValueBoundaryCondition(1),
                                                north = GradientBoundaryCondition(2),
                                                bottom = NoFluxBoundaryCondition(),
                                                top = ValueBoundaryCondition(-1))

    time_dependent_bcs = FieldBoundaryConditions(grid, ccc;
                                                 south = ValueBoundaryCondition(1),
                                                 top = ValueBoundaryCondition((x, y, t) -> t))

    c₁ = CenterField(grid; boundary_conditions=argument_free_bcs)
    c₂ = CenterField(grid; boundary_conditions=argument_free_bcs)
    c₃ = CenterField(grid; boundary_conditions=time_dependent_bcs)
    fields = (; c₁, c₂, c₃)

    for c in fields
        set!(c, (x, y, z) -> x + 2y + 3z)
    end

    fill_halo_regions!(c₁)
    fill_halo_regions!(c₂, clock, fields)
    @test Array(parent(c₁)) == Array(parent(c₂))

    fill_halo_regions!(c₃, clock, fields)
    Nz = size(grid, 3)
    top_interior = Array(interior(c₃, :, :, Nz))
    top_halo = Array(view(parent(c₃), :, :, Nz + 1 + grid.Hz))[1+grid.Hx:end-grid.Hx, 1+grid.Hy:end-grid.Hy]
    @test top_halo ≈ 2 * clock.time .- top_interior
end
