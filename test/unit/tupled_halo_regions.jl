include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition, MixedBoundaryCondition
using Oceananigans.Fields: tupled_fillable, halo_fill_groups, select_fields

function tupled_halo_test_fields(grid)
    TX, TY, TZ = topology(grid)
    x_bounded = TX === Bounded
    y_bounded = TY === Bounded

    lateral_center_bcs = x_bounded ? (west = ValueBoundaryCondition(2), east = GradientBoundaryCondition(-1)) : (;)
    meridional_center_bcs = y_bounded ? (south = MixedBoundaryCondition(1, 2), north = ValueBoundaryCondition(-3)) : (;)
    lateral_face_bcs = x_bounded ? (west = ImpenetrableBoundaryCondition(), east = NormalFlowBoundaryCondition(1)) : (;)

    ccc = (Center(), Center(), Center())
    value_bcs = FieldBoundaryConditions(grid, ccc; bottom = GradientBoundaryCondition(-2), top = ValueBoundaryCondition(1),
                                        lateral_center_bcs..., meridional_center_bcs...)

    u_bcs = FieldBoundaryConditions(grid, (Face(), Center(), Center()); lateral_face_bcs...)
    w_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom = ImpenetrableBoundaryCondition(),
                                    top = NormalFlowBoundaryCondition(0.5))
    time_dependent_bcs = FieldBoundaryConditions(grid, ccc; top = ValueBoundaryCondition((x, y, t) -> t))

    return (c₁ = CenterField(grid),
            c₂ = CenterField(grid; boundary_conditions = value_bcs),
            c₃ = CenterField(grid, Float32; boundary_conditions = value_bcs),
            c₄ = CenterField(grid),
            u₁ = XFaceField(grid; boundary_conditions = u_bcs),
            u₂ = XFaceField(grid),
            v = YFaceField(grid),
            w = ZFaceField(grid; boundary_conditions = w_bcs),
            ζ = Field{Face, Face, Center}(grid),
            time_dependent = CenterField(grid; boundary_conditions = time_dependent_bcs),
            reduced = Field{Center, Center, Nothing}(grid),
            windowed = CenterField(grid; indices = (:, :, 2:3)))
end

function set_random_interiors!(fields, seed)
    Random.seed!(seed)
    for field in fields
        set!(field, rand(size(field)...))
    end
    return nothing
end

halo_test_grid(arch, topology) = RectilinearGrid(arch; topology, size = (4, 5, 6), halo = (2, 3, 2),
                                                 x = (0, 1), y = (0, 2), z = [0, 0.5, 1, 2, 2.5, 3, 4])

@testset "Tupled halo filling for tuples of fields" begin
    topologies = ((Periodic, Periodic, Bounded),
                  (Periodic, Bounded, Bounded),
                  (Bounded, Periodic, Bounded),
                  (Bounded, Bounded, Bounded))

    @testset "Tupled fills are bitwise per-field fills [$(typeof(arch)), $topology]" for arch in archs, topology in topologies
        grid = halo_test_grid(arch, topology)
        clock = Clock(grid)
        clock.time = 2

        for kwargs in ((;), (; fill_normal_flow_bcs = false))
            individual = tupled_halo_test_fields(grid)
            tupled = tupled_halo_test_fields(grid)
            set_random_interiors!(individual, 1)
            set_random_interiors!(tupled, 1)

            for field in individual
                fill_halo_regions!(field, clock, individual; kwargs...)
            end

            fill_halo_regions!(tupled, clock, tupled; kwargs...)

            for name in keys(individual)
                @test Array(parent(individual[name])) == Array(parent(tupled[name]))
            end
        end

        # Nested tuples are filled by recursion
        individual = tupled_halo_test_fields(grid)
        nested = tupled_halo_test_fields(grid)
        set_random_interiors!(individual, 2)
        set_random_interiors!(nested, 2)
        names = (:c₁, :u₁, :c₂, :w, :reduced)
        foreach(name -> fill_halo_regions!(individual[name]), names)
        fill_halo_regions!(((nested.c₁, nested.u₁), nested.c₂, (; nested.w, nested.reduced)))

        for name in names
            @test Array(parent(individual[name])) == Array(parent(nested[name]))
        end
    end

    @testset "Grouping fields by halo-filling kernels" begin
        grid = halo_test_grid(CPU(), (Periodic, Bounded, Bounded))
        fields = tupled_halo_test_fields(grid)

        @test tupled_fillable(fields.c₁)
        @test tupled_fillable(fields.c₃)
        @test tupled_fillable(fields.w)
        @test !tupled_fillable(fields.time_dependent)
        @test !tupled_fillable(fields.reduced)
        @test !tupled_fillable(fields.windowed)
        @test !tupled_fillable(parent(fields.c₁))

        tupled_fillable_fields = @inferred select_fields(tupled_fillable, values(fields))
        groups = @inferred halo_fill_groups(tupled_fillable_fields)
        @test length(tupled_fillable_fields) == 9
        @test length(groups) == 5
        @test groups[1] === (fields.c₁, fields.c₂, fields.c₃, fields.c₄)
        @test groups[2] === (fields.u₁, fields.u₂)

        halo_fill_groups(tupled_fillable_fields)
        @test (@allocated halo_fill_groups(tupled_fillable_fields)) == 0
    end

    @testset "Tupled halo filling does not allocate on CPU" begin
        grid = halo_test_grid(CPU(), (Periodic, Bounded, Bounded))
        fields = tupled_halo_test_fields(grid)
        tupled_fillable_fields = select_fields(tupled_fillable, values(fields))

        fill_halo_regions!(tupled_fillable_fields)
        @test (@allocated fill_halo_regions!(tupled_fillable_fields)) == 0
    end
end
