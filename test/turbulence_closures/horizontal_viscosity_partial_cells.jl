include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using LinearAlgebra: Diagonal, Symmetric, eigmax, eigvals
using Random: Random
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: peripheral_node
using Oceananigans.Operators: Vᶠᶜᶜ, Vᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, HorizontalDivergenceScalarDiffusivity

# Horizontal viscous tendencies Gᵤ = -∂ⱼτ₁ⱼ and Gᵥ = -∂ⱼτ₂ⱼ of (u, v)
function viscous_tendencies(grid, closure, u, v)
    fields = (; u, v, w=ZFaceField(grid))
    args = (closure, nothing, Clock(time=0), fields, nothing)
    Gu = compute!(Field(- KernelFunctionOperation{Face, Center, Center}(∂ⱼ_τ₁ⱼ, grid, args...)))
    Gv = compute!(Field(- KernelFunctionOperation{Center, Face, Center}(∂ⱼ_τ₂ⱼ, grid, args...)))
    return Gu, Gv
end

# The horizontal viscous operator of `closure` as a matrix A, and the cell volumes M at the velocity points.
#
# Stack the interior velocities into one vector U = [vec(u); vec(v)]. The viscosity ν is constant, so the
# viscous tendencies are linear in U:
#
#     [vec(Gu); vec(Gv)] = A U,    Gu = -∂ⱼτ₁ⱼ,  Gv = -∂ⱼτ₂ⱼ,
#
# i.e., A is the discrete version of the viscous operator acting on (u, v): ν ∇² for a Laplacian viscosity,
# -ν ∇⁴ for a biharmonic one, and so on for any closure whose stresses are linear in (u, v). We build A one column
# at a time: column n is the tendency produced by a velocity that is 1 at point n and 0 everywhere else. A contains
# only the viscous term (no pressure, free surface, Coriolis, or advection).
#
# M holds the volumes Vᶠᶜᶜ (u points) and Vᶜᶠᶜ (v points), so the discrete kinetic energy is E = ½ Uᵀ diag(M) U.
# A itself is not symmetric, because each row is divided by its own volume. The viscous operator is self-adjoint
# when the product diag(M) A is symmetric.
#
# Both A and M are restricted to velocities off walls and land. Those velocities are always zero, so they would only
# add zero eigenvalues.
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

# The largest energy growth rate and the spectral radius of the viscous operator of `closure`.
#
# Growth rate: with dU/dt = A U and E = ½ Uᵀ D U, where D = diag(M), the energy changes at the rate
#
#     dE/dt = ½ (dU/dt)ᵀ D U + ½ Uᵀ D (dU/dt) = ½ Uᵀ (Aᵀ D + D A) U.
#
# Since Dᵀ = D, only the symmetric part of D A affects the energy. Substituting W = D^½ U gives
#
#     E = ½ Wᵀ W,    dE/dt = Wᵀ [½(S + Sᵀ)] W,    S = D^½ A D^-½.
#
# So the largest relative rate (dE/dt) / E over all velocities is a Rayleigh quotient:
#
#     growth_rate = max over W of 2 Wᵀ [½(S + Sᵀ)] W / Wᵀ W = 2 λmax(½(S + Sᵀ)).
#
# A viscous operator should never add energy, so we need growth_rate ≤ 0. A positive value means some velocity field
# gains energy from the viscosity. When A is self-adjoint in the volume-weighted inner product, S is symmetric.
#
# Spectral radius: max |λ| over the eigenvalues λ of A. S is similar to A, so they have the same eigenvalues. For an
# explicit time stepper, Δt × (spectral radius) must stay within the stepper's stability region, so the largest stable
# Δt scales as 1 / (spectral radius).
function viscous_growth_rate_and_spectral_radius(grid, closure)
    A, M = viscous_operator_matrix(grid, closure)
    m = sqrt.(M)
    S = Diagonal(m) * A * Diagonal(1 ./ m)
    growth_rate = eigmax(Symmetric((S + S') / 2)) * 2
    return growth_rate, maximum(abs, eigvals(A))
end

function partial_cell_bottom(Nx, Ny, H; land=false)
    Random.seed!(1234)
    bottom = - 5 .- (H - 5) .* rand(Nx, Ny)  # depths between 5 m and H, in a single level of thickness H
    land && (bottom[rand(Nx, Ny) .< 0.25] .= 10)
    return PartialCellBottom(bottom; minimum_fractional_cell_height=1e-3)
end

# Random velocities, damped for `Nt` time steps by the viscosity `Closure(ν = Δ^order / 1000Δt)`, which is
# stable over a flat bottom
function damping_of_random_velocities(arch, ib, Closure, order; Nt=200)
    Nx = Ny = 16
    Δ, H = 10e3, 4000
    underlying_grid = RectilinearGrid(arch; size=(Nx, Ny, 1), x=(0, Nx*Δ), y=(0, Ny*Δ), z=(-H, 0), halo=(4, 4, 4),
                                      topology=(Periodic, Periodic, Bounded))
    grid = isnothing(ib) ? underlying_grid : ImmersedBoundaryGrid(underlying_grid, ib)
    Δt = 600
    closure = Closure(ν = Δ^order / 1000Δt)
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

# Horizontal viscosities, each with the order of its operator: rates are in units of ν / Δ^order
horizontal_viscosities = (("Laplacian",           HorizontalScalarDiffusivity,           2),
                          ("divergence damping",  HorizontalDivergenceScalarDiffusivity, 2),
                          ("biharmonic",          HorizontalScalarBiharmonicDiffusivity, 4))

@testset "Horizontal viscosity with partial cells" begin
    Nx = Ny = 8
    Δ, H = 10e3, 4000

    for (name, Closure, order) in horizontal_viscosities
        closure = Closure(ν = Δ^order)

        @testset "Horizontal $name viscosity: dissipation and spectral radius with PartialCellBottom over random depths" begin
            for topology in ((Periodic, Periodic, Bounded), (Bounded, Bounded, Bounded)), land in (false, true)
                underlying_grid = RectilinearGrid(CPU(); size=(Nx, Ny, 1), x=(0, Nx*Δ), y=(0, Ny*Δ), z=(-H, 0),
                                                  halo=(4, 4, 4), topology)
                grid = ImmersedBoundaryGrid(underlying_grid, partial_cell_bottom(Nx, Ny, H; land))

                _, flat_spectral_radius = viscous_growth_rate_and_spectral_radius(underlying_grid, closure)
                growth_rate, spectral_radius = viscous_growth_rate_and_spectral_radius(grid, closure)

                # Energy never grows, up to round-off,
                @test growth_rate ≤ 1e-12 * spectral_radius
                # and partial cells do not tighten the time-step limit of a flat bottom
                @test spectral_radius ≤ flat_spectral_radius * (1 + 1e-12)
            end
        end

        @testset "Horizontal $name viscosity: dissipation and spectral radius with PartialCellBottom in Flat directions" begin
            alternating_bottom(ξ) = isodd(floor(Int, ξ / Δ)) ? -5 : -H
            x_varying_grid = RectilinearGrid(CPU(); size=(Nx, 1), x=(0, Nx*Δ), z=(-H, 0), halo=(4, 4), topology=(Periodic, Flat, Bounded))
            y_varying_grid = RectilinearGrid(CPU(); size=(Ny, 1), y=(0, Ny*Δ), z=(-H, 0), halo=(4, 4), topology=(Flat, Periodic, Bounded))
            column_grid = RectilinearGrid(CPU(); size=1, z=(-H, 0), halo=4, topology=(Flat, Flat, Bounded))

            for (underlying_grid, bottom) in ((x_varying_grid, alternating_bottom), (y_varying_grid, alternating_bottom), (column_grid, -H/2))
                grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom; minimum_fractional_cell_height=1e-3))
                _, flat_spectral_radius = viscous_growth_rate_and_spectral_radius(underlying_grid, closure)
                growth_rate, spectral_radius = viscous_growth_rate_and_spectral_radius(grid, closure)
                @test growth_rate ≤ 1e-12 * spectral_radius
                @test spectral_radius ≤ flat_spectral_radius * (1 + 1e-12)
            end
        end

        @testset "Horizontal $name viscosity: uniform cell thickness reproduces the grid without partial cells" begin
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
            @testset "Horizontal $name viscosity: random velocities stay bounded over random depths [$(typeof(arch))]" begin
                flat_initial_speed, flat_final_speed = damping_of_random_velocities(arch, nothing, Closure, order)
                initial_speed, final_speed = damping_of_random_velocities(arch, partial_cell_bottom(16, 16, 4000), Closure, order)
                @test flat_final_speed < flat_initial_speed
                @test isfinite(final_speed)
                @test final_speed < initial_speed
            end
        end
    end
end
