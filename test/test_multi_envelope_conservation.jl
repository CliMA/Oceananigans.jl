include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization
using Oceananigans.Architectures: architecture
using Oceananigans.Models: ZStarCoordinate

# Impose a horizontally-uniform, depth-dependent static envelope Jacobian σᵉ(k) and the consistent
# physical resting depth h = Σ Δr σᵉ. With σᵉ ≡ 1 this is plain z-star; with σᵉ varying in k it
# exercises the depth-dependent metric — in particular the `∂t_σ` override, which is the load-bearing
# piece of the discrete geometric conservation law.
function set_uniform_envelope_profile!(grid, profile)
    Nz = size(grid, 3)
    Hz = grid.Hz
    FT = eltype(grid)
    arch = architecture(grid)

    zlen = Nz + 2Hz
    profile_parent = FT[profile(clamp(kp - Hz, 1, Nz)) for kp in 1:zlen]
    σ_column = reshape(on_architecture(arch, profile_parent), 1, 1, :)

    for σᵉ in (grid.z.σᶜᶜᵉ, grid.z.σᶠᶜᵉ, grid.z.σᶜᶠᵉ, grid.z.σᶠᶠᵉ)
        parent(σᵉ) .= σ_column
    end

    Δrᵏ(k) = grid.z.Δᵃᵃᶜ isa Number ? grid.z.Δᵃᵃᶜ : grid.z.Δᵃᵃᶜ[k]
    h = sum(Δrᵏ(k) * profile(k) for k in 1:Nz)

    for hᵃ in (grid.z.hᶜᶜ, grid.z.hᶠᶜ, grid.z.hᶜᶠ, grid.z.hᶠᶠ)
        fill!(hᵃ, FT(h))
    end

    return h
end

function run_multi_envelope_conservation(arch, profile; Ni=60, Δt=20.0)
    z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=11)))
    grid = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                           topology=(Bounded, Bounded, Bounded))

    set_uniform_envelope_profile!(grid, profile)

    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=10),
                                        tracers = (:c, :constant),
                                        buoyancy = nothing,
                                        timestepper = :SplitRungeKutta3,
                                        vertical_coordinate = ZStarCoordinate())

    cᵢ(x, y, z) = 1 + 0.1 * sin(2π * x / 1e5) * cos(2π * y / 1e5)
    ηᵢ(x, y) = 0.2 * exp(-((x - 5e4)^2 + (y - 5e4)^2) / 4e8)
    set!(model, c=cᵢ, constant=1, η=ηᵢ)

    ∫c = Field(Integral(model.tracers.c))
    compute!(∫c)
    ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]

    w  = model.velocities.w
    Nz = size(grid, 3)

    constant_ok = true
    content_ok  = true
    top_w_ok    = true

    for _ in 1:Ni
        time_step!(model, Δt)

        compute!(∫c)
        ∫cₙ = Array(interior(∫c, 1, 1, 1))[1]
        content_ok &= isapprox(∫cₙ, ∫c₀; rtol=1e-10)

        cst = Array(interior(model.tracers.constant))
        constant_ok &= maximum(abs, cst .- 1) < 1e-10

        top_w = maximum(abs, Array(interior(w, :, :, Nz+1)))
        top_w_ok &= top_w < 1e-10
    end

    return (; constant_ok, content_ok, top_w_ok)
end

@testset "MultiEnvelope tracer conservation (geometric conservation law)" begin
    for arch in test_architectures()
        @testset "baseline σᵉ = 1 (reduces to z-star)" begin
            r = run_multi_envelope_conservation(arch, k -> 1.0)
            @test r.constant_ok
            @test r.content_ok
            @test r.top_w_ok
        end

        @testset "depth-dependent σᵉ(k) (load-bearing ∂t_σ override)" begin
            # σᵉ ramps 0.6 → 1.5 with depth: positive, mean ≠ 1 (so h ≠ Lz), varies in k.
            profile(k) = 0.5 + 0.1 * k
            r = run_multi_envelope_conservation(arch, profile)
            @test r.constant_ok
            @test r.content_ok
            @test r.top_w_ok
        end
    end
end
