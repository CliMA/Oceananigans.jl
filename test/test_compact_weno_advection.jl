include("dependencies_for_runtests.jl")

using Random

using Oceananigans.Advection: CompactWENO,
                              compact_weno_coefficient_tables,
                              compute_compact_reconstruction!,
                              contains_compact_weno,
                              materialize_advection,
                              precompute_advection!

using Oceananigans.BoundaryConditions: fill_halo_regions!

test_profile(z) = sin(2π * z) + 0.4 * cos(6π * z + 0.7)
test_profile_antiderivative(z) = -cos(2π * z) / 2π + 0.4 * sin(6π * z + 0.7) / 6π

exact_cell_averages(zᶠ) =
    [(test_profile_antiderivative(zᶠ[j+1]) - test_profile_antiderivative(zᶠ[j])) / (zᶠ[j+1] - zᶠ[j])
     for j in 1:length(zᶠ)-1]

function exponentially_stretched_faces(N, total_ratio)
    s = log(total_ratio)
    ξ = range(0.0, 1.0, length=N+1)
    return (exp.(s .* ξ) .- 1.0) ./ (exp(s) - 1.0) .- 1.0
end

function reconstruct_profile(arch, zᶠ)
    N = length(zᶠ) - 1
    grid = RectilinearGrid(arch, size=N, z=zᶠ, topology=(Flat, Flat, Bounded))
    scheme = materialize_advection(CompactWENO(Float64), grid)
    c = CenterField(grid)
    set!(c, reshape(exact_cell_averages(zᶠ), 1, 1, N))
    fill_halo_regions!(c)
    compute_compact_reconstruction!(scheme, c)
    return scheme
end

@testset "CompactWENO construction and guards" begin
    scheme = CompactWENO()
    @test summary(scheme) == "CompactWENO{Float64, Nothing}(order=5)"
    @test scheme.horizontal_scheme isa WENO
    @test scheme.buffer_scheme isa Centered
    @test eltype(CompactWENO(Float32)) == Float32

    @test contains_compact_weno(scheme)
    @test contains_compact_weno(FluxFormAdvection(WENO(), WENO(), CompactWENO()))
    @test !contains_compact_weno(WENO())

    periodic_grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1),
                                    topology=(Periodic, Periodic, Periodic))
    @test_throws ArgumentError materialize_advection(CompactWENO(), periodic_grid)

    bounded_grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
    @test_throws ArgumentError NonhydrostaticModel(bounded_grid; advection=CompactWENO(), tracers=:c)
end

@testset "CompactWENO uniform-grid coefficients" begin
    grid = RectilinearGrid(size=(1, 1, 16), extent=(1, 1, 1))
    tables = compact_weno_coefficient_tables(grid, Float64)
    k = 8
    for table in (tables.left, tables.right)
        @test all(isapprox.(table.a[k, :], [2/3, 1/3, 2/3], atol=1e-12))
        @test all(isapprox.(table.γ[k, :], [1/6, 5/6, 1/6], atol=1e-12))
        @test all(isapprox.(table.optimal5[k, :], [1/5, 1/2, 3/10], atol=1e-12))
    end
end

for arch in archs
    @testset "CompactWENO reconstruction [$(typeof(arch))]" begin
        # a constant field must be reconstructed exactly by every row type
        grid = RectilinearGrid(arch, size=16, z=exponentially_stretched_faces(16, 5),
                               topology=(Flat, Flat, Bounded))
        scheme = materialize_advection(CompactWENO(Float64), grid)
        c = CenterField(grid)
        set!(c, 0.73)
        fill_halo_regions!(c)
        compute_compact_reconstruction!(scheme, c)
        for ĉ in scheme.reconstructed_variable
            @test all(isapprox.(Array(interior(ĉ, 1, 1, :)), 0.73, atol=1e-12))
        end

        # observed convergence ≥ 4 on an exponentially stretched column
        errors = Float64[]
        for N in (64, 128)
            zᶠ = exponentially_stretched_faces(N, 8)
            reconstruction = reconstruct_profile(arch, zᶠ)
            ĉ = Array(interior(reconstruction.reconstructed_variable.left, 1, 1, :))
            push!(errors, maximum(abs(ĉ[k] - test_profile(zᶠ[k])) for k in 10:N-10))
        end
        observed_order = log2(errors[1] / errors[2])
        @test observed_order ≥ 4
    end

    @testset "CompactWENO stability with sign-changing w [$(typeof(arch))]" begin
        # regression for the mixed-bias singularity: noisy initial conditions produce
        # small sign-flipping w; the fixed-bias solves must remain finite
        grid = RectilinearGrid(arch, topology=(Periodic, Bounded, Bounded),
                               size=(8, 8, 16), halo=(3, 3, 3),
                               x=(0, 1e6), y=(-5e5, 5e5), z=(-1e3, 0))
        model = HydrostaticFreeSurfaceModel(grid;
            coriolis = BetaPlane(latitude=-45),
            buoyancy = BuoyancyTracer(),
            tracers = (:b, :c),
            momentum_advection = WENO(order=5),
            tracer_advection = FluxFormAdvection(WENO(order=5), WENO(order=5), CompactWENO()),
            free_surface = SplitExplicitFreeSurface(grid; substeps=10))

        Random.seed!(42)
        Δb = 5e4 * 8e-8
        bᵢ(x, y, z) = 4e-6 * z + Δb * min(max(0, y / 5e4 + 1/2), 1) + 1e-2 * Δb * randn()
        cᵢ(x, y, z) = exp(-(z + 250)^2 / 2e4)
        set!(model, b=bᵢ, c=cᵢ)

        for n in 1:30
            time_step!(model, 600)
        end

        @test all(isfinite, Array(interior(model.tracers.b)))
        @test all(isfinite, Array(interior(model.tracers.c)))
        @test all(isfinite, Array(interior(model.velocities.u)))
    end
end
