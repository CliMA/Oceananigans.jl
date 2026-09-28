# Linear scores of C-grid Coriolis schemes for shallow water on a doubly periodic f-plane:
# energy neutrality, vorticity (PV) consistency, null modes, inertia–gravity dispersion, and a check that
# the Oceananigans implementation in oriented_coriolis.jl reproduces the matrix operator.

using Oceananigans
using LinearAlgebra
using Printf
using Random
using Statistics

Oceananigans.defaults.FloatType = Float64

include("oriented_coriolis.jl")

const four_point_weights = [(p, q, 1/4) for p in (-1//2, 1//2), q in (-1//2, 1//2)]

# u[i, j] at (i-½, j), v[i, j] at (i, j-½), η[i, j] at (i, j), ζ[i, j] at (i-½, j-½); Δ = 1
struct PeriodicGrid
    N :: Int
end

index(g::PeriodicGrid, i, j) = mod1(i, g.N) + (mod1(j, g.N) - 1) * g.N

# Interpolation from a staggered lattice to another one, with offsets (p, q) from target to source
function interpolation_matrix(g, weights, source_shift)
    N² = g.N^2
    M = zeros(N², N²)
    for j in 1:g.N, i in 1:g.N, (p, q, w) in weights
        si, sj = source_shift(i, j, p, q)
        M[index(g, i, j), index(g, si, sj)] += w
    end
    return M
end

v_to_u(i, j, p, q) = (Int(i - 1//2 + p), Int(j + q + 1//2))
u_to_v(i, j, p, q) = (Int(i + p + 1//2), Int(j - 1//2 + q))

function difference_matrices(g)
    N² = g.N^2
    δxᵘ = zeros(N², N²); δyᵛ = zeros(N², N²)   # centres → u, v
    δxᶜ = zeros(N², N²); δyᶜ = zeros(N², N²)   # u, v → centres
    δxᶻ = zeros(N², N²); δyᶻ = zeros(N², N²)   # v, u → corners
    for j in 1:g.N, i in 1:g.N
        n = index(g, i, j)
        δxᵘ[n, n] += 1; δxᵘ[n, index(g, i-1, j)] -= 1
        δyᵛ[n, n] += 1; δyᵛ[n, index(g, i, j-1)] -= 1
        δxᶜ[n, index(g, i+1, j)] += 1; δxᶜ[n, n] -= 1
        δyᶜ[n, index(g, i, j+1)] += 1; δyᶜ[n, n] -= 1
        δxᶻ[n, n] += 1; δxᶻ[n, index(g, i-1, j)] -= 1
        δyᶻ[n, n] += 1; δyᶻ[n, index(g, i, j-1)] -= 1
    end
    return (; δxᵘ, δyᵛ, δxᶜ, δyᶜ, δxᶻ, δyᶻ)
end

# Coriolis operator acting on q = (u, v)
function coriolis_operator(g, scheme; f=1.0, ε=1/8, η=1/32, weights=inner_corner_weights(ε, η))
    D = difference_matrices(g)
    N² = g.N^2
    Mvu = interpolation_matrix(g, four_point_weights, v_to_u)
    Muv = interpolation_matrix(g, four_point_weights, u_to_v)
    C = [zeros(N², N²) f*Mvu; -f*Muv zeros(N², N²)]
    if scheme == :oriented
        # grad A + curl* B with A = N (f Γ), B = - Nᵀ (f D); N couples each centre to the corners of its 4 × 4 stencil
        stencil = OrientedCoriolis(; weights)
        Nmatrix = zeros(N², N²)
        for j in 1:g.N, i in 1:g.N, (n, (a, b)) in enumerate(corner_offsets)
            Nmatrix[index(g, i, j), index(g, i+a, j+b)] += stencil.weights[n]
        end
        δyᶻᵘ = zeros(N², N²); δxᶻᵛ = zeros(N², N²)
        for j in 1:g.N, i in 1:g.N
            n = index(g, i, j)
            δyᶻᵘ[n, index(g, i, j+1)] += 1; δyᶻᵘ[n, n] -= 1
            δxᶻᵛ[n, index(g, i+1, j)] += 1; δxᶻᵛ[n, n] -= 1
        end
        Γ = hcat(-D.δyᶻ, D.δxᶻ)
        divergence = hcat(D.δxᶜ, D.δyᶜ)
        A = f * Nmatrix * Γ
        B = - f * Nmatrix' * divergence
        C .+= [D.δxᵘ * A - δyᶻᵘ * B; D.δyᵛ * A + δxᶻᵛ * B]
    end
    return C
end

function scores(scheme; N=16, f=1.0, kw...)
    g = PeriodicGrid(N)
    D = difference_matrices(g)
    C = coriolis_operator(g, scheme; f, kw...)
    divergence = hcat(D.δxᶜ, D.δyᶜ)
    curl = hcat(-D.δyᶻ, D.δxᶻ)

    energy_error = norm(C + C') / norm(C)

    σ = svdvals(vcat(C, divergence))
    null_modes = count(<(1e-10 * maximum(σ)), σ)

    Random.seed!(1)
    ψ = randn(N^2)
    q = vcat(-D.δyᶻ' * ψ, D.δxᶻ' * ψ)   # nondivergent: u = -δy ψ, v = δx ψ with ψ at corners
    @assert norm(divergence * q) < 1e-10 * norm(q)
    curl_ratio = norm(curl * (C * q)) / (f * norm(curl * q))

    return (; energy_error, null_modes, curl_ratio)
end

function shallow_water_operator(scheme; N=32, f=1.0, gH=1/16, kw...)
    g = PeriodicGrid(N)
    N² = N^2
    D = difference_matrices(g)
    C = coriolis_operator(g, scheme; f, kw...)
    return [C  [-gH .* D.δxᵘ; -gH .* D.δyᵛ];
            -D.δxᶜ  -D.δyᶜ  zeros(N², N²)]
end

# 3×3 symbol of the shallow-water operator from its action on plane waves, amplitudes at each variable's own point
function dispersion(L, n, m; N=32, f=1.0, gH=1/16)
    g = PeriodicGrid(N)
    N² = N^2
    k, l = 2π * n / N, 2π * m / N
    positions = ((i, j) -> (i - 1/2, j), (i, j) -> (i, j - 1/2), (i, j) -> (i, j))
    wave(field) = [exp(im * (k * positions[field](i, j)[1] + l * positions[field](i, j)[2])) for i in 1:N, j in 1:N][:]
    probe = index(g, 3, 5)
    symbol = zeros(ComplexF64, 3, 3)
    for s in 1:3
        q = zeros(ComplexF64, 3N²)
        q[(s-1)*N² .+ (1:N²)] .= wave(s)
        Lq = L * q
        for t in 1:3
            symbol[t, s] = Lq[(t-1)*N² + probe] / wave(t)[probe]
        end
    end
    λ = eigvals(symbol)
    ω = sort(real.(im .* λ))
    exact = sqrt(f^2 + gH * (k^2 + l^2))
    return (; ω₋ = ω[1], ω₊ = ω[3], exact, growth = maximum(real, λ))
end

schemes = (:enstrophy, :oriented)

@info "Linear scores on a 16×16 periodic f-plane"
for scheme in schemes
    s = scores(scheme)
    @info @sprintf("%-12s energy error ‖C + Cᵀ‖/‖C‖ = %.1e   null modes = %3d   curl ratio = %.2e",
                   scheme, s.energy_error, s.null_modes, s.curl_ratio)
end

@info "Inertia–gravity frequency at R_d = Δ/4 (f = 1): ω₊ / exact - 1, reciprocity (ω₊ + ω₋)/f, growth rate"
N = 32
for scheme in schemes
    @info "  $scheme"
    L = shallow_water_operator(scheme; N)
    for (name, n, m) in (("(π, 0)", N÷2, 0), ("(0, π)", 0, N÷2), ("(π, π)", N÷2, N÷2),
                         ("(π/2, π/2)", N÷4, N÷4), ("(π/2, -π/2)", N÷4, -N÷4), ("(π/4, 0)", N÷8, 0))
        d = dispersion(L, n, m; N)
        @info @sprintf("    %-12s ω₊ = %.3f  exact %.3f  error %+6.1f%%   ω₊ + ω₋ = %+.2e   growth %.1e",
                       name, d.ω₊, d.exact, 100 * (d.ω₊ / d.exact - 1), d.ω₊ + d.ω₋, d.growth)
    end
    errors = [dispersion(L, n, m; N) for n in -N÷2+1:N÷2, m in -N÷2+1:N÷2 if (n, m) != (0, 0)]
    @info @sprintf("    worst error over the resolved range %+6.1f%%, worst |ω₊ + ω₋| = %.2e, worst growth %.1e",
                   100 * minimum(d -> d.ω₊ / d.exact - 1, errors), maximum(d -> abs(d.ω₊ + d.ω₋), errors),
                   maximum(d -> d.growth, errors))
end

#####
##### Oceananigans implementation versus the matrix operator
#####

using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U
using Oceananigans.BoundaryConditions: fill_halo_regions!

function implementation_error(coriolis_scheme, matrix_scheme; N=16, f=1.0, kw...)
    grid = RectilinearGrid(size=(N, N, 1), x=(0, N), y=(0, N), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded), halo=(5, 5, 1))
    u, v = XFaceField(grid), YFaceField(grid)
    Random.seed!(2)
    set!(u, randn(N, N, 1)); set!(v, randn(N, N, 1))
    fill_halo_regions!((u, v))
    coriolis = FPlane(f=f, scheme=coriolis_scheme)
    U = (u, v)
    Gu = [- x_f_cross_U(i, j, 1, grid, coriolis, U) for i in 1:N, j in 1:N][:]
    Gv = [- y_f_cross_U(i, j, 1, grid, coriolis, U) for i in 1:N, j in 1:N][:]
    C = coriolis_operator(PeriodicGrid(N), matrix_scheme; f, kw...)
    expected = C * vcat(interior(u, :, :, 1)[:], interior(v, :, :, 1)[:])
    return norm(vcat(Gu, Gv) - expected) / norm(expected)
end

@info @sprintf("Oceananigans OrientedCoriolis vs matrix: relative error %.1e", implementation_error(OrientedCoriolis(), :oriented))

#####
##### Energy neutrality on a sphere with land and variable f
#####

using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, mask_immersed_field!
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶠᶜᶜ, Δxᶜᶠᶜ, Δyᶜᶠᶜ, Δzᶠᶜᶜ, Δzᶜᶠᶜ

# EnergyConserving is skew for the weights Δx Δy Δz, which differ from the exact spherical cell volumes
function spherical_energy_error(scheme; Nx=72, Ny=40)
    underlying_grid = LatitudeLongitudeGrid(size=(Nx, Ny, 1), longitude=(0, 360), latitude=(-70, 70), z=(-1000, 0), halo=(5, 5, 1))
    Random.seed!(5)
    land = rand(Nx, Ny) .< 0.15
    land[:, 1:4] .= true
    land[:, end-3:end] .= true
    bottom = [land[i, j] ? 0.0 : -1000.0 for i in 1:Nx, j in 1:Ny]
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
    u, v = XFaceField(grid), YFaceField(grid)
    set!(u, randn(size(u))); set!(v, randn(size(v)))
    interior(v)[:, 1, :] .= 0
    interior(v)[:, Ny+1, :] .= 0
    mask_immersed_field!((u, v))
    fill_halo_regions!((u, v))
    coriolis = HydrostaticSphericalCoriolis(scheme=scheme)
    U = (u, v)
    work = 0.0
    scale = 0.0
    for j in 1:Ny+1, i in 1:Nx
        active_u = !immersed_peripheral_node(i, j, 1, grid, Face(), Center(), Center())
        active_v = !immersed_peripheral_node(i, j, 1, grid, Center(), Face(), Center())
        wu = (j ≤ Ny) * active_u * Δxᶠᶜᶜ(i, j, 1, grid) * Δyᶠᶜᶜ(i, j, 1, grid) * Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1] * (- x_f_cross_U(i, j, 1, grid, coriolis, U))
        wv = active_v * Δxᶜᶠᶜ(i, j, 1, grid) * Δyᶜᶠᶜ(i, j, 1, grid) * Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1] * (- y_f_cross_U(i, j, 1, grid, coriolis, U))
        work += wu + wv
        scale += abs(wu) + abs(wv)
    end
    return work / scale
end

@info "Coriolis work Σ ΔxΔyΔz u·(-f×u) / Σ |ΔxΔyΔz u·(-f×u)| on a 72×40 sphere with random land"
for scheme in (EnstrophyConserving(), EnergyConserving(), OrientedCoriolis())
    @info @sprintf("  %-20s %.1e", summary(scheme), spherical_energy_error(scheme))
end
@eval coastal_corner(i, j, k, grid) = false
@info @sprintf("  OrientedCoriolis with the coastal pairs kept %.1e", Base.invokelatest(spherical_energy_error, OrientedCoriolis()))
@eval coastal_corner(i, j, k, grid) = peripheral_node(i, j, k, grid, Face(), Face(), Center()) |
                                      inactive_node(i, j, k, grid, Face(), Face(), Center())

#####
##### Consistency near coastlines: a smooth free-slip flow in a basin whose coasts lie on grid lines
#####

using Oceananigans.Grids: λnode, φnode, peripheral_node, inactive_node
using Oceananigans.Operators: Δyᶠᶜᶜ, Δxᶜᶠᶜ

const basin = (λ = (20, 100), φ = (8, 48))

# Sum of basin modes: vanishes on the coast with a finite tangential velocity there, as a free-slip flow
function streamfunction(λ, φ)
    x = (λ - basin.λ[1]) / (basin.λ[2] - basin.λ[1])
    y = (φ - basin.φ[1]) / (basin.φ[2] - basin.φ[1])
    inside = (0 ≤ x ≤ 1) & (0 ≤ y ≤ 1)
    return inside * (sinpi(x) * sinpi(y) + sinpi(2x) * sinpi(3y) / 2)
end

function coastal_consistency(resolution; U₀=0.1)
    Nx, Ny = round(Int, 120 / resolution), round(Int, 56 / resolution)
    underlying_grid = LatitudeLongitudeGrid(size=(Nx, Ny, 1), longitude=(0, 120), latitude=(0, 56), z=(-1000, 0),
                                            halo=(5, 5, 1), topology=(Bounded, Bounded, Bounded))
    λc = λnodes(underlying_grid, Center(), Center(), Center())
    φc = φnodes(underlying_grid, Center(), Center(), Center())
    wet(λ, φ) = (basin.λ[1] < λ < basin.λ[2]) & (basin.φ[1] < φ < basin.φ[2])
    bottom = [wet(λc[i], φc[j]) ? -1000.0 : 0.0 for i in 1:Nx, j in 1:Ny]
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    R = grid.underlying_grid.radius
    ψ = [U₀ * R * deg2rad(basin.λ[2] - basin.λ[1]) / π *
         streamfunction(λnode(i, j, 1, grid, Face(), Face(), Center()), φnode(i, j, 1, grid, Face(), Face(), Center()))
         for i in 1:Nx+1, j in 1:Ny+1]

    u, v = XFaceField(grid), YFaceField(grid)
    for j in 1:Ny, i in 1:Nx+1
        u[i, j, 1] = - (ψ[i, j+1] - ψ[i, j]) / Δyᶠᶜᶜ(i, j, 1, grid)
    end
    for j in 1:Ny+1, i in 1:Nx
        v[i, j, 1] = (ψ[i+1, j] - ψ[i, j]) / Δxᶜᶠᶜ(i, j, 1, grid)
    end
    fill_halo_regions!((u, v))
    U = (u, v)

    wet_u = [!(peripheral_node(i, j, 1, grid, Face(), Center(), Center()) | inactive_node(i, j, 1, grid, Face(), Center(), Center()))
             for i in 1:Nx, j in 1:Ny]
    speed = sqrt(mean(interior(u, 1:Nx, :, 1)[wet_u] .^ 2))
    λu = λnodes(grid, Face(), Center(), Center())
    φu = φnodes(grid, Center(), Center(), Center())
    coast_distance(λ, φ) = min(λ - basin.λ[1], basin.λ[2] - λ, φ - basin.φ[1], basin.φ[2] - φ)
    coastal = [wet_u[i, j] && coast_distance(λu[i], φu[j]) < 2resolution for i in 1:Nx, j in 1:Ny]
    open_ocean = [wet_u[i, j] && !coastal[i, j] for i in 1:Nx, j in 1:Ny]

    Ω = Oceananigans.defaults.planet_rotation_rate
    coriolis = HydrostaticSphericalCoriolis(scheme=OrientedCoriolis())
    function maximum_increment()
        increment = [oriented_increment_u(i, j, 1, grid, coriolis, U) / (2Ω * speed) for i in 1:Nx, j in 1:Ny]
        return (coast = maximum(abs, increment[coastal]), open = maximum(abs, increment[open_ocean]))
    end
    excluded = maximum_increment()
    all_pairs = with_no_slip_coastal_pairs(maximum_increment)
    return (; excluded, all_pairs)
end

# The alternative to dropping the coastal pairs: keep them, with the circulation of the masked velocities, which at a
# coastal corner is that of a no-slip wall
using Oceananigans.Operators: Ax_qᶜᶠᶜ, Ay_qᶠᶜᶜ
@inline no_slip_circulationᶠᶠᶜ(i, j, k, grid, u, v) = Ax_qᶜᶠᶜ(i, j, k, grid, v) - Ax_qᶜᶠᶜ(i-1, j, k, grid, v) -
                                                      Ay_qᶠᶜᶜ(i, j, k, grid, u) + Ay_qᶠᶜᶜ(i, j-1, k, grid, u)

function with_no_slip_coastal_pairs(compute)
    @eval circulationᶠᶠᶜ(i, j, k, grid, u, v) = no_slip_circulationᶠᶠᶜ(i, j, k, grid, u, v)
    @eval coastal_corner(i, j, k, grid) = false
    result = Base.invokelatest(compute)
    @eval circulationᶠᶠᶜ(i, j, k, grid, u, v) = δxᶠᶠᶜ(i, j, k, grid, Ax_qᶜᶠᶜ, v) - δyᶠᶠᶜ(i, j, k, grid, Ay_qᶠᶜᶜ, u)
    @eval coastal_corner(i, j, k, grid) = peripheral_node(i, j, k, grid, Face(), Face(), Center()) |
                                          inactive_node(i, j, k, grid, Face(), Face(), Center())
    return result
end

@info "Coriolis-pressure increment |ΔC_u| / (2Ω U_rms) for a smooth free-slip basin flow: max within 2Δ of the coast / elsewhere"
for resolution in (4, 2, 1, 1/2)
    e = coastal_consistency(resolution)
    @info @sprintf("  Δ = %4.2f°  coastal pairs excluded: coast %.2e  open %.2e    all pairs kept: coast %.2e  open %.2e",
                   resolution, e.excluded.coast, e.excluded.open, e.all_pairs.coast, e.all_pairs.open)
end

#####
##### Equivariance under the reflection across the equator
#####

# u(i, j) → u(i, Ny+1-j) and v(i, j) → -v(i, Ny+2-j) on a latitude-longitude grid that is symmetric about the equator,
# with land that is also symmetric; the continuous Coriolis term commutes with this reflection because f is odd in latitude
function equatorial_equivariance_error(scheme; Nx=36, Ny=20, Nz=1)
    underlying_grid = LatitudeLongitudeGrid(size=(Nx, Ny, Nz), longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0), halo=(5, 5, 5))
    Random.seed!(7)
    northern_land = rand(Nx, Ny ÷ 2) .< 0.15
    land = hcat(reverse(northern_land, dims=2), northern_land)
    bottom = [land[i, j] ? 0.0 : -1000.0 for i in 1:Nx, j in 1:Ny]
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

    coriolis = HydrostaticSphericalCoriolis(scheme=scheme)
    tendencies(u, v) = ([- x_f_cross_U(i, j, k, grid, coriolis, (u, v)) for i in 1:Nx, j in 1:Ny, k in 1:Nz],
                        [- y_f_cross_U(i, j, k, grid, coriolis, (u, v)) for i in 1:Nx, j in 1:Ny+1, k in 1:Nz])

    u, v = XFaceField(grid), YFaceField(grid)
    set!(u, randn(size(u))); set!(v, randn(size(v)))
    interior(v)[:, 1, :] .= 0
    interior(v)[:, Ny+1, :] .= 0
    mask_immersed_field!((u, v))
    fill_halo_regions!((u, v))

    ũ, ṽ = XFaceField(grid), YFaceField(grid)
    interior(ũ) .= reverse(interior(u), dims=2)
    interior(ṽ) .= .- reverse(interior(v), dims=2)
    fill_halo_regions!((ũ, ṽ))

    Gu, Gv = tendencies(u, v)
    G̃u, G̃v = tendencies(ũ, ṽ)
    mismatch = max(maximum(abs, G̃u .- reverse(Gu, dims=2)), maximum(abs, G̃v .+ reverse(Gv, dims=2)))
    return mismatch / max(maximum(abs, Gu), maximum(abs, Gv))
end

@info "Equivariance under the reflection across the equator: max |C(M q) - M C(q)| / max |C(q)|"
@info @sprintf("  EnergyConserving                      %.1e", equatorial_equivariance_error(EnergyConserving()))
@info @sprintf("  OrientedCoriolis                      %.1e", equatorial_equivariance_error(OrientedCoriolis()))
