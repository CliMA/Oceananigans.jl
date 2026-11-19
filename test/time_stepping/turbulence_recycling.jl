include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.BoundaryConditions: step_boundary_conditions!
using Oceananigans.Grids: halo_size
using Oceananigans.Models: boundary_total_area

boundary_normal_dimension(side) = side in (:west, :east) ? 1 : side in (:south, :north) ? 2 : 3

# One recycling step of a single field against the formula φᵇ = φᵉˣᵗ + damping * (φʳ - ⟨φʳ⟩), on a stretched grid
function test_recycled_plane(arch, FT, side, BoundaryConditionConstructor, averaging)
    faces = (FT[0, 1, 3, 4, 6, 9, 10], FT[0, 2, 3, 5, 6, 8], FT[0, 1, 3, 6, 7])
    grid = RectilinearGrid(arch, FT; size = length.(faces) .- 1, x = faces[1], y = faces[2], z = faces[3], topology = (Open, Open, Open))
    cpu_grid = on_architecture(CPU(), grid)

    dim = boundary_normal_dimension(side)
    inward = side in (:west, :south, :bottom) ? 1 : -1
    t₁, t₂ = filter(!=(dim), (1, 2, 3))
    cells = 2
    ξ = faces[dim]
    recycling_distance = inward > 0 ? ξ[1 + cells] - ξ[1] : ξ[end] - ξ[end - cells]

    # A boundary-normal velocity, or a field at centers in the normal direction and at faces along the boundary
    normal_flow = BoundaryConditionConstructor === NormalFlowBoundaryCondition
    loc = ntuple(d -> d == (normal_flow ? dim : t₁) ? Face() : Center(), 3)
    N = size(grid, loc)
    H = halo_size(grid)

    φᵉˣᵗ = FT(0.7)
    damping = on_architecture(arch, FT[(i + 2j) / 10 for i in 1:N[t₁], j in 1:N[t₂]])
    scheme = TurbulenceRecycling(FT; recycling_distance, averaging, damping)
    boundary_conditions = FieldBoundaryConditions(grid, loc; side => BoundaryConditionConstructor(φᵉˣᵗ; scheme))
    field = Field(loc, grid; boundary_conditions)
    set!(field, (x, y, z) -> sin(x) + cos(2y) + z^2 / 10 + x * y * z / 50)

    boundary_index = inward > 0 ? (normal_flow ? 1 : 0) : size(grid, dim) + 1
    recycling_index = boundary_index + inward * cells
    plane(data, index) = view(selectdim(data, dim, index + H[dim]), (1:N[t₁]) .+ H[t₁], (1:N[t₂]) .+ H[t₂])

    before = Array(parent(field))
    step_boundary_conditions!((field,), Clock(time = zero(FT)), (;), one(FT))
    after = Array(parent(field))

    φʳ = plane(before, recycling_index)

    recycled_mean = if isnothing(averaging)
        zero(FT)
    else
        spacings = (Δx, Δy, Δz)
        point(i, j) = ntuple(d -> d == dim ? recycling_index : d == t₁ ? i : j, 3)
        weights = [prod(spacings[d](point(i, j)..., cpu_grid, loc...) for d in (t₁, t₂) if d in averaging) for i in 1:N[t₁], j in 1:N[t₂]]
        reduced_dimensions = Tuple(n for (n, d) in enumerate((t₁, t₂)) if d in averaging)
        sum(weights .* φʳ; dims = reduced_dimensions) ./ sum(weights; dims = reduced_dimensions)
    end

    @test plane(after, boundary_index) ≈ φᵉˣᵗ .+ Array(damping) .* (φʳ .- recycled_mean)

    # Nothing but the boundary plane changes
    plane(after, boundary_index) .= plane(before, boundary_index)
    @test after == before

    return nothing
end

# A pattern with the period of the recycling distance is advected through an open domain as through a periodic one
function test_recycled_tracer_matches_periodic(arch, FT, timestepper)
    N = 8
    advection = UpwindBiased(order = 1)
    periodic_grid = RectilinearGrid(arch, FT; size = (N, 2), x = (0, 1), z = (0, 1), topology = (Periodic, Flat, Bounded))
    open_grid = RectilinearGrid(arch, FT; size = (2N, 2), x = (0, 2), z = (0, 1), topology = (Open, Flat, Bounded))

    recycling = TurbulenceRecycling(FT; recycling_distance = 1, averaging = nothing)
    boundary_conditions = (u = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(1), east = NormalFlowBoundaryCondition(1)),
                           c = FieldBoundaryConditions(west = ValueBoundaryCondition(0; scheme = recycling)))

    periodic_model = NonhydrostaticModel(periodic_grid; timestepper, advection, tracers = :c)
    recycled_model = NonhydrostaticModel(open_grid; timestepper, advection, tracers = :c, boundary_conditions)

    for model in (periodic_model, recycled_model)
        set!(model, u = 1, c = (x, z) -> sin(2π * x) * (1 + z))

        for _ in 1:20
            time_step!(model, FT(1 / 4N))
        end
    end

    @test Array(interior(recycled_model.tracers.c))[1:N, :, :] ≈ Array(interior(periodic_model.tracers.c))

    return nothing
end

function test_recycled_inflow_model(arch, FT)
    N = 8
    grid = RectilinearGrid(arch, FT; size = (2N, N, N), x = (0, 2N), y = (0, N), z = (0, N), topology = (Open, Periodic, Bounded))
    Hx, Hy, Hz = halo_size(grid)

    Q = boundary_total_area(:west, grid)
    recycling(; kwargs...) = TurbulenceRecycling(FT; recycling_distance = N, kwargs...)

    u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(1; scheme = recycling(averaging = 2, target_transport = Q)),
                                    east = NormalFlowBoundaryCondition(1; scheme = PerturbationAdvection(FT)))
    w_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(0; scheme = recycling(averaging = (2, 3))))
    c_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(1; scheme = recycling(averaging = nothing, damping = (y, z, t) -> z / N)))

    model = NonhydrostaticModel(grid; tracers = :c, boundary_conditions = (u = u_bcs, w = w_bcs, c = c_bcs))

    set!(model, u = (x, y, z) -> 1 + sin(2π * y / N) * sin(π * z / N) / 10,
                w = (x, y, z) -> sin(2π * x / N) * sin(2π * y / N) * sin(π * z / N) / 10,
                c = (x, y, z) -> sin(2π * x / N) + cos(2π * y / N))

    for _ in 1:3
        time_step!(model, FT(1 / 20))
    end

    u, v, w = model.velocities
    c = model.tracers.c

    δ = Field(∂x(u) + ∂y(v) + ∂z(w))
    @test maximum(abs, δ) < sqrt(eps(FT))

    west_transport = Field(Integral(view(u, 1, :, :), dims = (2, 3)))
    @test Array(interior(west_transport))[1, 1, 1] ≈ Q

    # The tracer halo holds the exterior value plus the damped image of the recycling plane
    z = reshape(Array(znodes(grid, Center())), 1, N)
    c_halo = Array(parent(c))[Hx, (1:N) .+ Hy, (1:N) .+ Hz]
    @test c_halo ≈ 1 .+ z ./ N .* Array(interior(c))[N, :, :]

    # A face on the bottom wall receives no fluctuation
    w_halo = Array(parent(w))[Hx, (1:N) .+ Hy, 1 + Hz]
    @test all(w_halo .== 0)

    return nothing
end

@testset "Turbulence recycling [$(summary(arch)), $FT]" for arch in archs, FT in float_types
    @testset "Recycled plane on the $side boundary" for side in (:west, :east, :south, :north, :bottom, :top)
        t₁, t₂ = filter(!=(boundary_normal_dimension(side)), (1, 2, 3))
        test_recycled_plane(arch, FT, side, NormalFlowBoundaryCondition, t₁)
        test_recycled_plane(arch, FT, side, ValueBoundaryCondition, (t₁, t₂))
    end

    test_recycled_plane(arch, FT, :west, NormalFlowBoundaryCondition, 3)
    test_recycled_plane(arch, FT, :west, ValueBoundaryCondition, nothing)

    @testset "Recycled tracer with $timestepper" for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        test_recycled_tracer_matches_periodic(arch, FT, timestepper)
    end

    test_recycled_inflow_model(arch, FT)
end
