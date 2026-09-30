#
# The Jiang & Shu profile of `one_dimensional_advection.jl` advected past a staircase immersed boundary.
#
# The velocity comes from a streamfunction that vanishes on and inside the staircase, so it is divergence free to
# roundoff and tangent to the boundary: any new extremum is made by the reconstruction, not by the flow.
#

using Oceananigans
using Oceananigans.Advection: div_Uc, GhostCells
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, ZeroField
using Oceananigans.ImmersedBoundaries: immersed_cell
using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Utils: NormalDivision
using Printf

const Lx, Lz = 1.0, 1.0

@inline G(x, β, z) = exp(-β * (x - z)^2)
@inline F(x, α, a) = sqrt(max(1 - α^2 * (x - a)^2, 0.0))

@inline function jiang_shu(x; Z = -0.7, δ = 0.005, β = log(2) / (36δ^2), a = 0.5, α = 10)
    -0.8 ≤ x ≤ -0.6 && return (G(x, β, Z - δ) + 4G(x, β, Z) + G(x, β, Z + δ)) / 6
    -0.4 ≤ x ≤ -0.2 && return 1.0
     0.0 ≤ x ≤  0.2 && return 1.0 - abs(10 * (x - 0.1))
     0.4 ≤ x ≤  0.6 && return (F(x, α, a - δ) + 4F(x, α, a) + F(x, α, a + δ)) / 6
    return 0.0
end

function staircase_grid(; Nx, Nz)
    underlying = RectilinearGrid(size = (Nx, Nz), halo = (6, 6), x = (0, Lx), z = (-Lz, 0),
                                 topology = (Periodic, Flat, Bounded))
    bottom(x) = 0.35Lx < x < 0.65Lx ? -0.35Lz : -Lz
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))
end

function tangent_flow(grid; A = 0.05)
    Nx, _, Nz = size(grid)
    Δx, Δz = Lx / Nx, Lz / Nz
    dry(i, k) = k < 1 || k > Nz || immersed_cell(mod1(i, Nx), 1, k, grid)

    ψ = zeros(Nx + 1, Nz + 1)
    for i in 1:Nx+1, k in 1:Nz+1
        solid = dry(i-1, k-1) | dry(i, k-1) | dry(i-1, k) | dry(i, k)
        ψ[i, k] = solid ? 0.0 : A * sin(2π * (i - 1) * Δx / Lx) * sin(π * (k - 1) * Δz / Lz)
    end

    u, w = XFaceField(grid), ZFaceField(grid)
    for i in 1:Nx, k in 1:Nz;   u[i, 1, k] = -(ψ[i, k+1] - ψ[i, k]) / Δz; end
    for i in 1:Nx, k in 1:Nz+1; w[i, 1, k] =  (ψ[i+1, k] - ψ[i, k]) / Δx; end
    fill_halo_regions!(u)
    fill_halo_regions!(w)

    return (; u, v = ZeroField(), w)
end

function advect(grid, U, scheme; Δt, Nsteps)
    Nx, _, Nz = size(grid)
    c = CenterField(grid)
    set!(c, (x, z) -> jiang_shu(2z + 1))
    fill_halo_regions!(c)
    c¹, c², Gc = CenterField(grid), CenterField(grid), CenterField(grid)

    wet = [!immersed_cell(i, 1, k, grid) for i in 1:Nx, k in 1:Nz]
    moment(φ, p) = sum(wet[i, k] ? φ[i, 1, k]^p * Vᶜᶜᶜ(i, 1, k, grid) : 0.0 for i in 1:Nx, k in 1:Nz)

    tendency!(φ) = for i in 1:Nx, k in 1:Nz
        Gc[i, 1, k] = wet[i, k] ? -div_Uc(i, 1, k, grid, scheme, U, φ) : 0.0
    end

    stage!(out, a, b, γ) = (for i in 1:Nx, k in 1:Nz
        out[i, 1, k] = wet[i, k] ? γ * a[i, 1, k] + (1 - γ) * (b[i, 1, k] + Δt * Gc[i, 1, k]) : a[i, 1, k]
    end; fill_halo_regions!(out))

    mass, variance = moment(c, 1), moment(c, 2)
    cmin, cmax = Inf, -Inf

    for n in 1:Nsteps
        tendency!(c);  stage!(c¹, c, c,  0)
        tendency!(c¹); stage!(c², c, c¹, 3/4)
        tendency!(c²); stage!(c,  c, c², 1/3)
        wet_values = [c[i, 1, k] for i in 1:Nx, k in 1:Nz if wet[i, k]]
        cmin = min(cmin, minimum(wet_values))
        cmax = max(cmax, maximum(wet_values))
    end

    return cmin, cmax, 1 - moment(c, 2) / variance, (moment(c, 1) - mass) / mass
end

function run_hat_advection(FT = Float64; Nx = 128, Nz = 64, courant = 0.2, Nsteps = 3000, order = 7)
    grid = staircase_grid(; Nx, Nz)
    U = tangent_flow(grid)
    Δt = courant / (maximum(abs, interior(U.u)) * Nx / Lx + maximum(abs, interior(U.w)) * Nz / Lz)

    weno(boundary_scheme) = WENO(FT; order, weight_computation = NormalDivision, boundary_scheme)

    schemes = ["Centered(order=2)"          => weno(Centered(FT; order = 2)),
               "UpwindBiased(order=1)"      => weno(UpwindBiased(FT; order = 1)),
               "GhostCells()"               => weno(GhostCells(FT)),
               "GhostCells(monotone=false)" => weno(GhostCells(FT; monotone = false))]

    println(rpad("scheme", 28), rpad("min", 12), rpad("max", 12), rpad("variance lost", 16), "mass drift")
    for (name, scheme) in schemes
        cmin, cmax, variance_lost, mass_drift = advect(grid, U, scheme; Δt, Nsteps)
        @printf("%-28s%-12.5f%-12.5f%-16s%.2e\n", name, cmin, cmax, @sprintf("%.3f%%", 100variance_lost), mass_drift)
    end
end

run_hat_advection()
