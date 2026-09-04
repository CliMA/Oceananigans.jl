#
# The Jiang & Shu profile of `one_dimensional_advection.jl` -- Gaussian, top hat, triangle, semi-ellipse --
# advected past a staircase immersed boundary.
#
# The velocity comes from a streamfunction that vanishes on and inside the staircase, so it is divergence
# free to roundoff and exactly tangent to the boundary: every face on the staircase carries zero velocity, and
# `∇·u = 0` in every wet cell. Any new extremum is therefore made by the reconstruction, not by the flow.
#
# The top hat is the case of interest: a discontinuity carried along the wall, where the reconstruction has
# lost part of its stencil and cannot appeal to smoothness.
#

using Oceananigans
using Oceananigans.Advection: div_Uc
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interior, ZeroField
using Oceananigans.Grids: xnode, znode, Center, Face
using Oceananigans.ImmersedBoundaries: immersed_cell, peripheral_node
using Oceananigans.Operators: div_xyᶜᶜᶜ, δzᵃᵃᶜ, Az_qᶜᶜᶠ, V⁻¹ᶜᶜᶜ, Vᶜᶜᶜ
using Oceananigans.Utils: NormalDivision
using Printf
using JLD2

const Lx, Lz = 1.0, 1.0

#####
##### Initial condition: the profile of `one_dimensional_advection.jl`, mapped from x ∈ [-1, 1] onto the column
#####

@inline G(x, β, z) = exp(-β * (x - z)^2)
@inline F(x, α, a) = sqrt(max(1 - α^2 * (x - a)^2, 0.0))

const Zc = -0.7
const δc = 0.005
const βc = log(2) / (36 * δc^2)
const ac = 0.5
const αc = 10

@inline function jiang_shu(x)
    if x <= -0.6 && x >= -0.8
        return (G(x, βc, Zc - δc) + 4G(x, βc, Zc) + G(x, βc, Zc + δc)) / 6
    elseif x <= -0.2 && x >= -0.4
        return 1.0
    elseif x <= 0.2 && x >= 0.0
        return 1.0 - abs(10 * (x - 0.1))
    elseif x <= 0.6 && x >= 0.4
        return (F(x, αc, ac - δc) + 4F(x, αc, ac) + F(x, αc, ac + δc)) / 6
    else
        return 0.0
    end
end

initial_tracer(x, z) = jiang_shu(2z + 1)

#####
##### Staircase grid and a flow that is divergence free and tangent to it
#####

function staircase_grid(; Nx = 128, Nz = 64)
    underlying = RectilinearGrid(size = (Nx, Nz), halo = (6, 6),
                                 x = (0, Lx), z = (-Lz, 0), topology = (Periodic, Flat, Bounded))
    step(x) = (0.35Lx < x < 0.65Lx) ? -0.35Lz : -Lz
    bottom = Field{Center, Center, Nothing}(underlying)
    set!(bottom, step)
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))
end

function tangent_flow(grid; A = 0.05)
    Nx, _, Nz = size(grid)
    underlying = grid.underlying_grid
    dry(i, k) = (k < 1 || k > Nz) ? true : immersed_cell(mod1(i, Nx), 1, k, grid)

    # ψ at (Face, Center, Face), zeroed on every node touching a dry cell and at the surface
    ψ = zeros(Nx + 1, Nz + 1)
    for i in 1:Nx+1, k in 1:Nz+1
        x = xnode(i, underlying, Face())
        z = znode(k, underlying, Face())
        solid = dry(i-1, k-1) | dry(i, k-1) | dry(i-1, k) | dry(i, k)
        ψ[i, k] = solid ? 0.0 : A * sin(2π * x / Lx) * sin(π * (z + Lz) / Lz)
    end

    u = XFaceField(grid)
    w = ZFaceField(grid)
    Δx = Lx / Nx
    Δz = Lz / Nz
    for i in 1:Nx, k in 1:Nz;   u[i, 1, k] = -(ψ[i, k+1] - ψ[i, k]) / Δz; end
    for i in 1:Nx, k in 1:Nz+1; w[i, 1, k] =  (ψ[i+1, k] - ψ[i, k]) / Δx; end
    fill_halo_regions!(u)
    fill_halo_regions!(w)

    return (; u, v = ZeroField(), w)
end

divergence(i, j, k, grid, u, v, w) =
    div_xyᶜᶜᶜ(i, j, k, grid, u, v) + V⁻¹ᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, w)

function verify_flow(grid, U)
    Nx, _, Nz = size(grid)
    worst_divergence = 0.0
    worst_wall_velocity = 0.0
    for i in 1:Nx, k in 1:Nz
        immersed_cell(i, 1, k, grid) ||
            (worst_divergence = max(worst_divergence, abs(divergence(i, 1, k, grid, U.u, U.v, U.w))))
        peripheral_node(i, 1, k, grid, Face(), Center(), Center()) &&
            (worst_wall_velocity = max(worst_wall_velocity, abs(U.u[i, 1, k])))
        peripheral_node(i, 1, k, grid, Center(), Center(), Face()) &&
            (worst_wall_velocity = max(worst_wall_velocity, abs(U.w[i, 1, k])))
    end
    return worst_divergence, worst_wall_velocity
end

#####
##### SSP-RK3 advection
#####

function advect(grid, U, scheme; Δt, Nsteps, snapshot_interval = 0)
    Nx, _, Nz = size(grid)
    c  = CenterField(grid); set!(c, initial_tracer); fill_halo_regions!(c)
    c1 = CenterField(grid); c2 = CenterField(grid); G = CenterField(grid)

    wet = [!immersed_cell(i, 1, k, grid) for i in 1:Nx, k in 1:Nz]
    V   = [Vᶜᶜᶜ(i, 1, k, grid) for i in 1:Nx, k in 1:Nz]

    moment(cc, p) = sum(wet[i, k] ? cc[i, 1, k]^p * V[i, k] : 0.0 for i in 1:Nx, k in 1:Nz)

    tendency!(cc) = for i in 1:Nx, k in 1:Nz
        G[i, 1, k] = wet[i, k] ? -div_Uc(i, 1, k, grid, scheme, U, cc) : 0.0
    end

    stage!(out, a, b, α) = (for i in 1:Nx, k in 1:Nz
        out[i, 1, k] = wet[i, k] ? α * a[i, 1, k] + (1 - α) * (b[i, 1, k] + Δt * G[i, 1, k]) : a[i, 1, k]
    end; fill_halo_regions!(out))

    mass₀ = moment(c, 1)
    var₀  = moment(c, 2)
    lo, hi = Inf, -Inf

    snapshots = Matrix{Float64}[]
    snapshot_times = Float64[]
    take_snapshot(n) = (push!(snapshots, [c[i, 1, k] for i in 1:Nx, k in 1:Nz]);
                        push!(snapshot_times, n * Δt))
    snapshot_interval > 0 && take_snapshot(0)

    for n in 1:Nsteps
        tendency!(c);  stage!(c1, c, c,  0.0)
        tendency!(c1); stage!(c2, c, c1, 3/4)
        tendency!(c2); stage!(c,  c, c2, 1/3)
        values = [c[i, 1, k] for i in 1:Nx, k in 1:Nz if wet[i, k]]
        lo = min(lo, minimum(values))
        hi = max(hi, maximum(values))
        snapshot_interval > 0 && n % snapshot_interval == 0 && take_snapshot(n)
    end

    return (min = lo, max = hi,
            variance_lost = 1 - moment(c, 2) / var₀,
            mass_drift = (moment(c, 1) - mass₀) / mass₀,
            snapshots = snapshots, snapshot_times = snapshot_times, wet = wet)
end

function run_hat_advection(; Nx = 128, Nz = 64, courant = 0.2, Nsteps = 3000,
                             save_prefix = nothing, snapshot_interval = 20)
    grid = staircase_grid(; Nx, Nz)
    U = tangent_flow(grid)
    d, wv = verify_flow(grid, U)
    @printf("staircase %d×%d:  max |∇·u| = %.2e   max |u| on the wall = %.2e\n", Nx, Nz, d, wv)

    Δt = courant / (maximum(abs, interior(U.u)) / (Lx / Nx) + maximum(abs, interior(U.w)) / (Lz / Nz))
    @printf("Courant %.2f, Δt = %.3e, %d steps\n\n", courant, Δt, Nsteps)

    schemes = ["WENO(order=7)"             => WENO(order = 7, weight_computation = NormalDivision),
               "WENO(order=5)"             => WENO(order = 5, weight_computation = NormalDivision),
               "WENO(order=7) min_buffer=1" => WENO(order = 7, weight_computation = NormalDivision,
                                                    minimum_buffer_upwind_order = 1)]

    interval = isnothing(save_prefix) ? 0 : snapshot_interval

    println(rpad("scheme", 34), rpad("min", 12), rpad("max", 12), rpad("variance lost", 16), "mass drift")
    results = Dict{String, Any}()
    for (name, scheme) in schemes
        r = advect(grid, U, scheme; Δt, Nsteps, snapshot_interval = interval)
        @printf("%-34s%-12.5f%-12.5f%-16s%.2e\n", name, r.min, r.max,
                @sprintf("%.3f%%", 100r.variance_lost), r.mass_drift)
        results[name] = r
    end

    if !isnothing(save_prefix)
        file = save_prefix * ".jld2"
        jldopen(file, "w") do f
            f["Nx"] = Nx; f["Nz"] = Nz; f["Lx"] = Lx; f["Lz"] = Lz; f["Δt"] = Δt
            f["wet"] = first(values(results)).wet
            f["initial"] = first(values(results)).snapshots[1]
            for (name, r) in results
                g = "schemes/" * name
                f[g * "/snapshots"] = r.snapshots
                f[g * "/times"]     = r.snapshot_times
                f[g * "/min"]       = r.min
                f[g * "/max"]       = r.max
                f[g * "/variance_lost"] = r.variance_lost
            end
        end
        println("\nwrote $file  ($(length(first(values(results)).snapshots)) snapshots per scheme)")
    end

    return results
end

# `julia immersed_hat_advection.jl` prints the table only.
# `julia immersed_hat_advection.jl mylabel` also dumps snapshots to mylabel.jld2 for the movie script.
run_hat_advection(; save_prefix = isempty(ARGS) ? nothing : ARGS[1])
