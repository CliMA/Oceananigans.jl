include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using LinearAlgebra: Diagonal, Symmetric, diag, eigmax, eigvals
using Random: Random
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: peripheral_node
using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ

# Horizontal viscous tendencies Gᵤ = -∂ⱼτ₁ⱼ and Gᵥ = -∂ⱼτ₂ⱼ of (u, v)
function viscous_tendencies(grid, closure, u, v)
    fields = (; u, v, w=ZFaceField(grid))
    args = (closure, nothing, Clock(time=0), fields, nothing)
    Gu = compute!(Field(- KernelFunctionOperation{Face, Center, Center}(∂ⱼ_τ₁ⱼ, grid, args...)))
    Gv = compute!(Field(- KernelFunctionOperation{Center, Face, Center}(∂ⱼ_τ₂ⱼ, grid, args...)))
    return Gu, Gv
end

# The viscous operator as a matrix A, restricted to velocities off walls and land, and the volumes M at those velocities.
function viscous_operator_matrix(grid, closure)
    u, v = XFaceField(grid), YFaceField(grid)
    Nu, Nv = length(interior(u)), length(interior(v))
    A = zeros(Nu + Nv, Nu + Nv)

    for n in 1:Nu+Nv
        U = zeros(Nu + Nv)
        U[n] = 1
        set!(u, reshape(U[1:Nu], size(interior(u))))
        set!(v, reshape(U[Nu+1:end], size(interior(v))))
        fill_halo_regions!((u, v))
        Gu, Gv = viscous_tendencies(grid, closure, u, v)
        A[:, n] = vcat(vec(interior(Gu)), vec(interior(Gv)))
    end

    Vu = compute!(Field(KernelFunctionOperation{Face, Center, Center}(Vᶠᶜᶜ, grid)))
    Vv = compute!(Field(KernelFunctionOperation{Center, Face, Center}(Vᶜᶠᶜ, grid)))
    M = vcat(vec(interior(Vu)), vec(interior(Vv)))

    active(ℓx, ℓy, f) = vec([!peripheral_node(i, j, k, grid, ℓx, ℓy, Center()) for (i, j, k) in Tuple.(CartesianIndices(interior(f)))])
    a = vcat(active(Face(), Center(), u), active(Center(), Face(), v))

    return A[a, a], M[a]
end

# Largest rate of change of ½ Σ V |u|² relative to ½ Σ V |u|² over all u (dissipative if ≤ 0), and the spectral radius
function biharmonic_growth_rate_and_spectral_radius(grid, closure)
    A, M = viscous_operator_matrix(grid, closure)
    m = sqrt.(M)
    S = Diagonal(m) * A * Diagonal(1 ./ m) # similar to A, and symmetric when A is self-adjoint in the volume-weighted norm
    growth_rate = eigmax(Symmetric((S + S') / 2)) * 2
    return growth_rate, maximum(abs, eigvals(A))
end

function partial_cell_bottom(Nx, Ny, H; land=false)
    Random.seed!(1234)
    bottom = - 5 .- (H - 5) .* rand(Nx, Ny)  # depths between 5 m and H, in a single level of thickness H
    land && (bottom[rand(Nx, Ny) .< 0.25] .= 10)
    return PartialCellBottom(bottom; minimum_fractional_cell_height=1e-3)
end

# Random velocities, damped for `Nt` time steps by a biharmonic viscosity that is stable over a flat bottom
function biharmonic_damping_of_random_velocities(arch, ib; Nt=200)
    Nx = Ny = 16
    Δ, H = 10e3, 4000
    underlying_grid = RectilinearGrid(arch; size=(Nx, Ny, 1), x=(0, Nx*Δ), y=(0, Ny*Δ), z=(-H, 0), halo=(4, 4, 4),
                                      topology=(Periodic, Periodic, Bounded))
    grid = isnothing(ib) ? underlying_grid : ImmersedBoundaryGrid(underlying_grid, ib)
    Δt = 600
    closure = HorizontalScalarBiharmonicDiffusivity(ν = Δ^4 / 1000Δt)
    model = HydrostaticFreeSurfaceModel(grid; closure, momentum_advection=nothing, tracers=(), buoyancy=nothing,
                                        free_surface=ImplicitFreeSurface())
    Random.seed!(42)
    set!(model, u=(x, y, z) -> randn(), v=(x, y, z) -> randn())
    u₀ = maximum(abs, model.velocities.u)
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    return u₀, maximum(abs, model.velocities.u)
end

@testset "Horizontal biharmonic viscosity with partial cells" begin
    Nx = Ny = 8
    Δ, H = 10e3, 4000
    closure = HorizontalScalarBiharmonicDiffusivity(ν = Δ^4) # rates are in units of ν / Δ⁴

    @testset "Dissipation and spectral radius with PartialCellBottom over random depths" begin
        for topology in ((Periodic, Periodic, Bounded), (Bounded, Bounded, Bounded)), land in (false, true)
            @info "  Testing biharmonic dissipation with PartialCellBottom [$topology, land=$land]..."
            underlying_grid = RectilinearGrid(CPU(); size=(Nx, Ny, 1), x=(0, Nx*Δ), y=(0, Ny*Δ), z=(-H, 0),
                                              halo=(4, 4, 4), topology)
            grid = ImmersedBoundaryGrid(underlying_grid, partial_cell_bottom(Nx, Ny, H; land))

            flat_growth_rate, flat_spectral_radius = biharmonic_growth_rate_and_spectral_radius(underlying_grid, closure)
            growth_rate, spectral_radius = biharmonic_growth_rate_and_spectral_radius(grid, closure)

            # Energy never grows, up to round-off,
            @test growth_rate ≤ 1e-12 * spectral_radius
            # and partial cells do not tighten the time-step limit of a flat bottom
            @test spectral_radius ≤ flat_spectral_radius * (1 + 1e-12)
        end
    end

    @testset "Uniform cell thickness reproduces the grid without partial cells" begin
        underlying_grid = RectilinearGrid(CPU(); size=(Nx, Ny, 1), x=(0, Nx*Δ), y=(0, Ny*Δ), z=(-H, 0),
                                          halo=(4, 4, 4), topology=(Periodic, Periodic, Bounded))
        grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(-H))

        Random.seed!(7)
        uᵢ, vᵢ = randn(Nx, Ny, 1), randn(Nx, Ny, 1)
        G = map((underlying_grid, grid)) do g
            u, v = XFaceField(g), YFaceField(g)
            set!(u, uᵢ)
            set!(v, vᵢ)
            fill_halo_regions!((u, v))
            Array.(interior.(viscous_tendencies(g, closure, u, v)))
        end

        scale = maximum(abs, G[1][1])
        @test maximum(abs, G[2][1] .- G[1][1]) ≤ 1e-12 * scale
        @test maximum(abs, G[2][2] .- G[1][2]) ≤ 1e-12 * scale
    end

    for arch in archs
        @testset "Random velocities stay bounded over random depths [$(typeof(arch))]" begin
            @info "  Testing biharmonic damping with PartialCellBottom [$(typeof(arch))]..."
            u₀, u_flat = biharmonic_damping_of_random_velocities(arch, nothing)
            u₀, u_partial = biharmonic_damping_of_random_velocities(arch, partial_cell_bottom(16, 16, 4000))
            @test u_flat < u₀
            @test isfinite(u_partial)
            @test u_partial < u₀
        end
    end
end
