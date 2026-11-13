include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: Closed, Connected, Open, TwoSided
using Oceananigans.Operators: ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ
using Oceananigans.TurbulenceClosures: DynamicSmagorinsky, LagrangianAveraging
using Oceananigans.TurbulenceClosures.Smagorinskys: filterᶜᶜᶜ, filterᶠᶜᶜ, filterᶜᶠᶜ, filterᶜᶜᶠ,
                                                    test_filteredᶜ, test_filteredᶠ, ᾱ², compute_test_filtered!,
                                                    ΣᵢⱼΣᵢⱼᶜᶜᶜ, Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ

Random.seed!(1234)

const ccc = (Center, Center, Center)

const located_filters = ((filterᶜᶜᶜ, ccc, CenterField),
                         (filterᶠᶜᶜ, (Face, Center, Center), XFaceField),
                         (filterᶜᶠᶜ, (Center, Face, Center), YFaceField),
                         (filterᶜᶜᶠ, (Center, Center, Face), ZFaceField))

# Two-point averages to cell corners and back, which the test filter equals away from boundaries
@inline composite_filter(i, j, k, grid, c) = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, c)

"Evaluate the kernel function `f`, located at `loc`, over the interior of `grid`."
evaluate(f, loc, grid, args...) = Array(interior(compute!(Field(KernelFunctionOperation{loc...}(f, grid, args...)))))

"""
Return the response of `filterᶜᶜᶜ` to an impulse at the center of `grid`,
and the tensor product of (¼, ½, ¼) that it should equal.
"""
function impulse_response(grid)
    FT = eltype(grid)
    center = size(grid) .÷ 2 .+ 1
    impulse = zeros(FT, size(grid)...)
    impulse[center...] = 1

    c = CenterField(grid)
    set!(c, impulse)
    fill_halo_regions!(c)
    response = evaluate(filterᶜᶜᶜ, ccc, grid, c)

    # Flat directions have a single point, which keeps all of the weight
    offset_range(N) = N == 1 ? (0:0) : (-1:1)
    weight(offset, N) = N == 1 ? one(FT) : FT[1/4, 1/2, 1/4][offset + 2]
    expected = zeros(FT, size(grid)...)
    for offsets in Iterators.product(offset_range.(size(grid))...)
        expected[(center .+ offsets)...] = prod(weight.(offsets, size(grid)))
    end

    return response, expected
end

"Write `NaN` into the halo points of `field` that `fill_halo_regions!` does not write."
function poison_unfilled_halos!(field)
    data = parent(field)
    original = Array(data)

    fill!(data, NaN)
    fill!(interior(field), 1)
    fill_halo_regions!(field)
    unfilled = isnan.(Array(data))

    original[unfilled] .= NaN
    copyto!(data, original)

    return nothing
end

@testset "Dynamic Smagorinsky test filter stencil [$(typeof(arch)), $FT]" for arch in archs, FT in float_types
    grid = RectilinearGrid(arch, FT; size=(5, 5, 5), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))

    # 27 points with weights 1/8 (center), 1/16 (faces), 1/32 (edges) and 1/64 (corners)
    response, expected = impulse_response(grid)
    @test response == expected
    @test count(!iszero, response) == 27
    @test extrema(response[response .> 0]) == (1/64, 1/8)

    # Away from boundaries the filter is ℑxyzᶜᶜᶜ(ℑxyzᶠᶠᶠ(f)) at every location
    for (located_filter, loc, LocatedField) in located_filters
        q = LocatedField(grid)
        set!(q, rand(FT, size(q)...))
        fill_halo_regions!(q)
        @test evaluate(located_filter, loc, grid, q) ≈ evaluate(composite_filter, loc, grid, q)
    end

    # Nothing is filtered along Flat directions, which leaves 9 points
    flat_grid = RectilinearGrid(arch, FT; size=(5, 5), extent=(1, 1), topology=(Periodic, Flat, Periodic))
    response, expected = impulse_response(flat_grid)
    @test response == expected
    @test count(!iszero, response) == 9
end

@testset "Dynamic Smagorinsky test filter at boundaries [$(typeof(arch)), $FT]" for arch in archs, FT in float_types
    grid = RectilinearGrid(arch, FT; size=(4, 5, 6), extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))

    # A quantity that varies along one direction is filtered along it, except at the cell centers next
    # to the boundaries and at the faces on them. NaNs in the halos show that nothing beyond is read.
    for (located_filter, loc, LocatedField) in located_filters, dim in 1:3
        q = LocatedField(grid)
        fill!(parent(q), NaN)

        N = size(q, dim)
        profile = rand(FT, N)
        shape = ntuple(d -> d == dim ? N : 1, 3)
        set!(q, reshape(profile, shape) .* ones(FT, size(q)...))

        filtered_profile = copy(profile)
        @. filtered_profile[2:N-1] = (profile[1:N-2] + 2 * profile[2:N-1] + profile[3:N]) / 4

        @test all(evaluate(located_filter, loc, grid, q) .≈ reshape(filtered_profile, shape))
    end

    # The filter-width ratio follows the number of directions that are filtered at each cell
    Nx, Ny, Nz = size(grid)
    @test ᾱ²(2, 2, 2, grid) isa FT
    @test ᾱ²(2, 2, 2, grid) == 4
    @test ᾱ²(1, 2, 2, grid) ≈ ᾱ²(Nx, 2, 2, grid) ≈ ᾱ²(2, 1, 2, grid) ≈ ᾱ²(2, Ny, 2, grid) ≈ FT(4)^FT(2/3)
    @test ᾱ²(2, 2, 1, grid) ≈ ᾱ²(2, 2, Nz, grid) ≈ FT(4)^FT(2/3)
    @test ᾱ²(1, 1, 2, grid) ≈ ᾱ²(Nx, 2, Nz, grid) ≈ FT(4)^FT(1/3)
    @test ᾱ²(1, 1, 1, grid) == ᾱ²(Nx, Ny, Nz, grid) == 1

    periodic_grid = RectilinearGrid(arch, FT; size=(4, 5, 6), extent=(1, 1, 1), topology=(Periodic, Periodic, Periodic))
    @test ᾱ²(1, 1, 1, periodic_grid) == ᾱ²(Nx, Ny, Nz, periodic_grid) == 4

    flat_grid = RectilinearGrid(arch, FT; size=(4, 6), extent=(1, 1), topology=(Periodic, Flat, Bounded))
    @test ᾱ²(1, 1, 2, flat_grid) == 4
    @test ᾱ²(1, 1, 1, flat_grid) == ᾱ²(1, 1, size(flat_grid, 3), flat_grid) == 2
end

@testset "Dynamic Smagorinsky test filter and grid topology" begin
    N = 4

    # A side whose halos hold boundary data stops the filter; periodic and connected sides do not
    for (topology, bounded_on_the_left, bounded_on_the_right) in ((Periodic,                     false, false),
                                                                  (Bounded,                      true,  true),
                                                                  (TwoSided{Open, Open},         true,  true),
                                                                  (TwoSided{Closed, Open},       true,  true),
                                                                  (TwoSided{Closed, Connected},  true,  false),
                                                                  (TwoSided{Connected, Open},    false, true),
                                                                  (TwoSided{Connected, Connected}, false, false))

        filteredᶜ(i) = !(bounded_on_the_left && i ≤ 1) && !(bounded_on_the_right && i ≥ N)
        filteredᶠ(i) = !(bounded_on_the_left && i ≤ 1) && !(bounded_on_the_right && i ≥ N + 1)

        @test all(test_filteredᶜ(i, topology, N) == filteredᶜ(i) for i in 0:N+1)
        @test all(test_filteredᶠ(i, topology, N) == filteredᶠ(i) for i in 0:N+2)
    end

    @test !test_filteredᶜ(1, Flat, 1)
    @test !test_filteredᶠ(1, Flat, 1)
end

@testset "Dynamic Smagorinsky test-filtered strain rate next to walls [$(typeof(arch))]" for arch in archs
    # Uniform shear between no-slip walls. The test filter leaves a linear velocity profile unchanged,
    # so that the test-filtered strain rate equals the strain rate up to the walls: ΣᵢⱼΣᵢⱼ = γ² / 2.
    γ = 2
    grid = RectilinearGrid(arch; size=(4, 4, 6), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Periodic, Periodic, Bounded))

    no_slip = FieldBoundaryConditions(grid, (Face(), Center(), Center());
                                      bottom = ValueBoundaryCondition(0),
                                      top = ValueBoundaryCondition(γ))

    u = XFaceField(grid; boundary_conditions=no_slip)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    set!(u, (x, y, z) -> γ * z)
    fill_halo_regions!((u, v, w))

    @test all(evaluate(ΣᵢⱼΣᵢⱼᶜᶜᶜ, ccc, grid, u, v, w) .≈ γ^2 / 2)

    ū, v̄, w̄ = XFaceField(grid), YFaceField(grid), ZFaceField(grid)
    compute_test_filtered!(ū, filterᶠᶜᶜ, u)
    compute_test_filtered!(v̄, filterᶜᶠᶜ, v)
    compute_test_filtered!(w̄, filterᶜᶜᶠ, w)
    @test all(evaluate(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ, ccc, grid, ū, v̄, w̄) .≈ γ^2 / 2)
end

@testset "Dynamic Smagorinsky in a closed domain [$(typeof(arch)), $name averaging]" for arch in archs,
        (name, averaging) in (("Lagrangian", LagrangianAveraging()), ("directional", (1, 2)))

    grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1), topology=(Bounded, Bounded, Bounded))
    closure = DynamicSmagorinsky(; averaging)

    # `fill_halo_regions!` writes one halo layer beyond a `Bounded` side, and nothing where two such
    # sides meet. The solution must not depend on the halo points that are left unfilled.
    reference_model = NonhydrostaticModel(grid; closure)
    poisoned_model = NonhydrostaticModel(grid; closure)
    initial_velocities = map(velocity -> rand(size(velocity)...), reference_model.velocities)

    for model in (reference_model, poisoned_model)
        set!(model; initial_velocities...)
    end

    # The stored test-filtered velocities are the test filter of the velocities
    closure_fields = reference_model.closure_fields
    filtered_velocities = (closure_fields.ū, closure_fields.v̄, closure_fields.w̄)
    for (f̄, (located_filter, loc, _), velocity) in zip(filtered_velocities, located_filters[2:4], reference_model.velocities)
        @test Array(interior(f̄)) == evaluate(located_filter, loc, grid, velocity)
    end

    foreach(poison_unfilled_halos!, poisoned_model.velocities)

    for _ in 1:3
        time_step!(reference_model, 1e-3)
        time_step!(poisoned_model, 1e-3)
    end

    for (reference, poisoned) in zip(reference_model.velocities, poisoned_model.velocities)
        @test all(isfinite, Array(interior(poisoned)))
        @test Array(interior(reference)) == Array(interior(poisoned))
    end

    for (reference, poisoned) in zip(reference_model.closure_fields, poisoned_model.closure_fields)
        reference isa Field || continue
        @test all(isfinite, Array(interior(poisoned)))
        @test Array(interior(reference)) == Array(interior(poisoned))
    end

    # Opposite walls are treated alike: the mirror image of a flow has the mirror image of LM and MM
    if averaging isa Tuple
        u₀, v₀, w₀ = initial_velocities
        model = reference_model
        LM = model.closure_fields.LM
        MM = model.closure_fields.MM

        set!(model; initial_velocities...)
        reference_LM = Array(interior(LM))
        reference_MM = Array(interior(MM))

        set!(model, u = -reverse(u₀, dims=1), v = reverse(v₀, dims=1), w = reverse(w₀, dims=1))
        @test Array(interior(LM)) ≈ reverse(reference_LM, dims=1)
        @test Array(interior(MM)) ≈ reverse(reference_MM, dims=1)

        set!(model, u = reverse(u₀, dims=3), v = reverse(v₀, dims=3), w = -reverse(w₀, dims=3))
        @test Array(interior(LM)) ≈ reverse(reference_LM, dims=3)
        @test Array(interior(MM)) ≈ reverse(reference_MM, dims=3)
    end
end

@testset "Dynamic Smagorinsky with a Flat direction [$(typeof(arch))]" for arch in archs
    grid = RectilinearGrid(arch; size=(4, 4), extent=(1, 1), topology=(Periodic, Flat, Bounded))
    model = NonhydrostaticModel(grid; closure=DynamicSmagorinsky())
    set!(model, u = rand(size(model.velocities.u)...), w = rand(size(model.velocities.w)...))

    for _ in 1:3
        time_step!(model, 1e-3)
    end

    @test all(isfinite, Array(interior(model.closure_fields.νₑ)))
    @test all(isfinite, Array(interior(model.velocities.u)))
end
