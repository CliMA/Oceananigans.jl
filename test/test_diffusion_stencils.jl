include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models: ZStarCoordinate, ZCoordinate, surface_kernel_parameters
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _update_zstar_scaling!
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity,
                                       TriadIsopycnalSkewSymmetricDiffusivity,
                                       DiffusiveFormulation, AdvectiveFormulation
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂xᵣᶠᶜᶜ, Axᶠᶜᶜ, Vᶜᶜᶜ

# Time-stepping an isopycnal closure on a z-star grid produces no NaN.
function isopycnal_closure_zstar_produces_no_nans(arch, closure)
    H = 50.0
    Lx = Ly = 100.0
    z = MutableVerticalDiscretization((-H, 0))

    grid = RectilinearGrid(arch; size=(4, 4, 8),
                           x=(0, Lx), y=(0, Ly), z=z,
                           topology=(Periodic, Periodic, Bounded))

    free_surface = SplitExplicitFreeSurface(grid; substeps=10)

    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface,
                                        vertical_coordinate = ZStarCoordinate(),
                                        buoyancy = BuoyancyTracer(),
                                        closure = closure,
                                        tracers = :b)

    # Sinusoidal perturbation to create non-zero slopes and drive dynamics
    set!(model, b = (x, y, z) -> 1e-5 * z + 1e-7 * sin(2π * x / Lx))

    for n in 1:10
        time_step!(model, 1.0)
    end

    return !any(isnan, model.tracers.b)
end

# Static and mutable grids produce the same results when using ZCoordinate
# (no grid evolution), verifying that r-based derivatives reduce to standard
# derivatives on non-deformed grids.
function isopycnal_static_and_mutable_grids_agree(arch, closure)
    Lx = Ly = 100.0
    H = 50.0

    static_grid = RectilinearGrid(arch; size=(4, 4, 8),
                                  x=(0, Lx), y=(0, Ly), z=(-H, 0),
                                  topology=(Periodic, Periodic, Bounded))

    mutable_grid = RectilinearGrid(arch; size=(4, 4, 8),
                                    x=(0, Lx), y=(0, Ly),
                                    z=MutableVerticalDiscretization((-H, 0)),
                                    topology=(Periodic, Periodic, Bounded))

    kw = (; free_surface = ImplicitFreeSurface(),
            vertical_coordinate = ZCoordinate(),
            buoyancy = BuoyancyTracer(),
            closure = closure,
            tracers = :b)

    static_model  = HydrostaticFreeSurfaceModel(static_grid; kw...)
    mutable_model = HydrostaticFreeSurfaceModel(mutable_grid; kw...)

    b_init(x, y, z) = 1e-5 * z + 1e-7 * sin(2π * x / Lx)
    set!(static_model,  b = b_init)
    set!(mutable_model, b = b_init)

    time_step!(static_model, 10.0)
    time_step!(mutable_model, 10.0)

    b_s = Array(interior(static_model.tracers.b))
    b_m = Array(interior(mutable_model.tracers.b))

    return all(b_s .≈ b_m)
end

function redi_preserves_stratification(arch)
    closure = IsopycnalSkewSymmetricDiffusivity(κ_skew = nothing, κ_symmetric = 1000.0)
    grid = RectilinearGrid(arch; size=(4, 4, 8),
                           x=(0, 100), y=(0, 100), z=(-100, 0),
                           topology=(Periodic, Periodic, Bounded))

    model = HydrostaticFreeSurfaceModel(grid;
                                        buoyancy = BuoyancyTracer(),
                                        closure = closure,
                                        tracers = :b)

    set!(model, b = (x, y, z) -> 1e-5 * z)

    b⁰ = Array(interior(model.tracers.b))
    time_step!(model, 10.0)
    bⁿ = Array(interior(model.tracers.b))

    return all(b⁰ .≈ bⁿ)
end

function redi_preserves_stratification_on_zstar(arch)
    H = 50.0
    z_faces = MutableVerticalDiscretization((-H / 1.3, 0))
    closure = IsopycnalSkewSymmetricDiffusivity(κ_skew = nothing, κ_symmetric = 1000.0)
    grid = RectilinearGrid(arch; size=(4, 4, 8),
                           x=(0, 100), y=(0, 100), z=z_faces,
                           topology=(Periodic, Periodic, Bounded))

    # Uniformly deform the grid: σ = 1.3 stretches vertical spacing
    σ = 1.3
    η = H * (1 - 1 / σ)
    fill!(grid.z.ηⁿ,   η)
    fill!(grid.z.σᶜᶜ⁻, σ)
    fill!(grid.z.σᶜᶜⁿ, σ)
    fill!(grid.z.σᶜᶠⁿ, σ)
    fill!(grid.z.σᶠᶠⁿ, σ)
    fill!(grid.z.σᶠᶜⁿ, σ)

    model = HydrostaticFreeSurfaceModel(grid;
                                        buoyancy = BuoyancyTracer(),
                                        closure = closure,
                                        tracers = :b)

    set!(model, b = (x, y, z) -> 1e-5 * z)

    b⁰ = Array(interior(model.tracers.b))
    time_step!(model, 10.0)
    bⁿ = Array(interior(model.tracers.b))

    return all(b⁰ .≈ bⁿ)
end

# Compute horizontal diffusion tendency using the r-derivative
@kernel function horizontal_r_flux_divergence!(∇F, grid, c)
    i, j, k = @index(Global, NTuple)
    Fe = Axᶠᶜᶜ(i+1, j, k, grid) * ∂xᵣᶠᶜᶜ(i+1, j, k, grid, c)
    Fw = Axᶠᶜᶜ(i,   j, k, grid) * ∂xᵣᶠᶜᶜ(i,   j, k, grid, c)
    @inbounds ∇F[i, j, k] = Fe - Fw
end

# Compute horizontal diffusion tendency using the z-derivative
@kernel function horizontal_z_flux_divergence!(∇F, grid, c)
    i, j, k = @index(Global, NTuple)
    Fe = Axᶠᶜᶜ(i+1, j, k, grid) * ∂xᶠᶜᶜ(i+1, j, k, grid, c)
    Fw = Axᶠᶜᶜ(i,   j, k, grid) * ∂xᶠᶜᶜ(i,   j, k, grid, c)
    @inbounds ∇F[i, j, k] = Fe - Fw
end

# Compute the variance tendency Σ c * ∇F.
function compute_variance_tendency(arch, grid, cf, kernel!)
    ∇F = CenterField(grid)
    launch!(arch, grid, :xyz, kernel!, ∇F, grid, cf)
    c = Array(interior(cf))
    G = Array(interior(∇F))
    return sum(c .* G)
end

# On a deformed z-star grid, horizontal diffusion using the z-derivative (∂x,
# with chain-rule interpolation of ∂z c) can PRODUCE variance. The ℑxz interpolation
# mixes ∂z c from columns with very different Δz, causing the chain-rule correction
# to overshoot and create anti-diffusion.
# The tracer profile is the eigenvector of the variance tendency matrix corresponding
# to its most positive eigenvalue (found by numerical eigenanalysis).
function horizontal_diffusion_variance(arch)
    H = 1.0
    Nx, Nz = 3, 4
    η₀ = [0.4, -0.4, 0.4]  # σ contrast
    z_ref = MutableVerticalDiscretization((-H, 0))

    grid = RectilinearGrid(arch; size=(Nx, Nz),
                           x=(0, Float64(Nx)), z=z_ref,
                           topology=(Periodic, Flat, Bounded))

    η = ZFaceField(grid; indices = (:, :, Nz+1))
    set!(η, η₀)
    fill_halo_regions!(η)

    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, η, grid)

    c₀ = [0.0168  -0.1936  -0.177   0.5344;
          0.0423  -0.1258  -0.4813  0.2036;
          0.0168  -0.1936  -0.177   0.5344]

    c = CenterField(grid)
    set!(c, c₀)
    fill_halo_regions!(c)

    dc²ᶻ = compute_variance_tendency(arch, grid, c, horizontal_z_flux_divergence!)
    dc²ʳ = compute_variance_tendency(arch, grid, c, horizontal_r_flux_divergence!)

    return dc²ᶻ > 0 && dc²ʳ < 0
end

@testset "Isopycnal closures with r-based derivatives" begin
    @info "Testing isopycnal closures with r-based derivatives..."

    for arch in archs
        issd_closures = [
            ("ISSD DiffusiveFormulation",
             IsopycnalSkewSymmetricDiffusivity(κ_skew=1000.0, κ_symmetric=1000.0,
                                               skew_flux_formulation=DiffusiveFormulation())),
            ("ISSD AdvectiveFormulation",
             IsopycnalSkewSymmetricDiffusivity(κ_skew=1000.0, κ_symmetric=1000.0,
                                               skew_flux_formulation=AdvectiveFormulation())),
            ("TISSD",
             TriadIsopycnalSkewSymmetricDiffusivity(κ_skew=1000.0, κ_symmetric=1000.0)),
        ]

        @testset "z-star time-stepping [$arch]" begin
            for (name, closure) in issd_closures
                @testset "$name on z-star" begin
                    @info "  Testing $name on z-star grid [$arch]..."
                    @test isopycnal_closure_zstar_produces_no_nans(arch, closure)
                end
            end
        end

        @testset "Static vs mutable grid agreement [$arch]" begin
            for (name, closure) in issd_closures
                @testset "$name" begin
                    @info "  Testing $name: static vs mutable grid [$arch]..."
                    @test isopycnal_static_and_mutable_grids_agree(arch, closure)
                end
            end
        end

        @testset "Redi preserves pure stratification [$arch]" begin
            @info "  Testing Redi preserves pure stratification [$arch]..."
            @test redi_preserves_stratification(arch)

            @info "  Testing Redi preserves pure stratification with zstar [$arch]..."
            @test redi_preserves_stratification_on_zstar(arch)
        end

        @testset "Horizontal diffusion variance on z-star [$arch]" begin
            @info "  Testing horizontal diffusion variance properties [$arch]..."
            @test horizontal_diffusion_variance(arch)
        end
    end
end
