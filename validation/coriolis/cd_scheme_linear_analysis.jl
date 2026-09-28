# Linear analysis of the C-D Coriolis scheme (Adcroft, Hill and Marshall, 1999) for shallow water on a C grid.
#
# 1. Eigenvalues of the semi-discrete operator on a doubly periodic equatorial β-plane, open or with land:
#    a positive real part is a growing mode of the spatial discretization.
# 2. Fourier symbol on an f-plane: damping rate of the physical modes versus wavelength,
#    expressed as an equivalent Laplacian and biharmonic viscosity.

using LinearAlgebra
using Printf

const day = 86400.0

#####
##### 1. Growing modes with land
#####

# u[i, j] at (i-½, j), v[i, j] at (i, j-½), η[i, j] at (i, j); unknowns u, v, η, uᴰ, vᴰ for scheme = :cd
function linear_operator(scheme, land; Nx=12, Ny=48, Δ=1e5, g=9.81, H=2.5^2/9.81, f₀=2*7.292e-5*sind(24), relaxation_time=Inf)
    N = Nx * Ny
    index(i, j) = mod1(i, Nx) + (mod1(j, Ny) - 1) * Nx
    is_land(i, j) = (mod1(i, Nx), mod1(j, Ny)) in land
    wet_u(i, j) = !(is_land(i, j) || is_land(i-1, j))
    wet_v(i, j) = !(is_land(i, j) || is_land(i, j-1))
    U, V, Hη, Uᴰ, Vᴰ = 0, N, 2N, 3N, 4N
    L = zeros(scheme == :cd ? 5N : 3N, scheme == :cd ? 5N : 3N)
    f(y) = f₀ * sin(2π * y / Ny)
    v_neighbors(i, j) = ((i-1, j), (i, j), (i-1, j+1), (i, j+1))
    u_neighbors(i, j) = ((i, j), (i+1, j), (i, j-1), (i+1, j-1))
    r = 1 / relaxation_time

    for j in 1:Ny, i in 1:Nx
        if wet_u(i, j)
            L[U + index(i, j), Hη + index(i, j)] -= g / Δ
            L[U + index(i, j), Hη + index(i-1, j)] += g / Δ
        end
        if wet_v(i, j)
            L[V + index(i, j), Hη + index(i, j)] -= g / Δ
            L[V + index(i, j), Hη + index(i, j-1)] += g / Δ
        end
        L[Hη + index(i, j), U + index(i+1, j)] -= H / Δ * wet_u(i+1, j)
        L[Hη + index(i, j), U + index(i, j)]   += H / Δ * wet_u(i, j)
        L[Hη + index(i, j), V + index(i, j+1)] -= H / Δ * wet_v(i, j+1)
        L[Hη + index(i, j), V + index(i, j)]   += H / Δ * wet_v(i, j)
    end

    G = L[1:3N, :]   # C-grid tendency without Coriolis

    for j in 1:Ny, i in 1:Nx
        fu, fv = f(j - 1/2), f(j - 1)
        if scheme == :energy
            wet_u(i, j) && for (ii, jj) in v_neighbors(i, j); L[U + index(i, j), V + index(ii, jj)] += f(jj - 1) / 4 * wet_v(ii, jj); end
            wet_v(i, j) && for (ii, jj) in u_neighbors(i, j); L[V + index(i, j), U + index(ii, jj)] -= f(jj - 1/2) / 4 * wet_u(ii, jj); end
        else
            if wet_u(i, j)
                L[U + index(i, j), Vᴰ + index(i, j)] += fu
                L[Vᴰ + index(i, j), U + index(i, j)] -= fu
                L[Vᴰ + index(i, j), Vᴰ + index(i, j)] -= r
                for (ii, jj) in v_neighbors(i, j)
                    L[Vᴰ + index(i, j), :] .+= G[V + index(ii, jj), :] ./ 4
                    L[Vᴰ + index(i, j), V + index(ii, jj)] += r / 4 * wet_v(ii, jj)
                end
            end
            if wet_v(i, j)
                L[V + index(i, j), Uᴰ + index(i, j)] -= fv
                L[Uᴰ + index(i, j), V + index(i, j)] += fv
                L[Uᴰ + index(i, j), Uᴰ + index(i, j)] -= r
                for (ii, jj) in u_neighbors(i, j)
                    L[Uᴰ + index(i, j), :] .+= G[U + index(ii, jj), :] ./ 4
                    L[Uᴰ + index(i, j), U + index(ii, jj)] += r / 4 * wet_u(ii, jj)
                end
            end
        end
    end

    return L
end

e_folding_days(L) = (rate = maximum(real, eigvals(L)); rate > 1e-10 ? 1 / rate / day : Inf)

lands = (open = Set{Tuple{Int, Int}}(),
         coast = Set((i, j) for i in 1:3, j in 1:48),
         equatorial_island = Set((i, j) for i in 5:7, j in 23:26),
         strait = union(Set((i, j) for i in 3:5, j in 20:28), Set((i, j) for i in 7:9, j in 20:28)))

@info "e-folding time [days] of the fastest growing mode (Inf = neutrally stable)"
for (name, land) in pairs(lands)
    energy = e_folding_days(linear_operator(:energy, land))
    cd = e_folding_days(linear_operator(:cd, land))
    cd10 = e_folding_days(linear_operator(:cd, land; relaxation_time=10day))
    cd30 = e_folding_days(linear_operator(:cd, land; relaxation_time=30day))
    @info @sprintf("%-18s EnergyConserving: %8.1f   CD τ=∞: %8.1f   CD τ=10d: %8.1f   CD τ=30d: %8.1f", name, energy, cd, cd10, cd30)
end

#####
##### 2. Implied viscosity on an f-plane
#####

function cd_symbol(k, l; Δ=1e5, f=1e-4, gH=2.5^2, g=9.81, relaxation_time=10day)
    H = gH / g
    Dx, Dy = 2im * sin(k * Δ / 2) / Δ, 2im * sin(l * Δ / 2) / Δ
    A = cos(k * Δ / 2) * cos(l * Δ / 2)
    r = 1 / relaxation_time
    return [ 0        0        -g*Dx      0      f;
             0        0        -g*Dy     -f      0;
            -H*Dx    -H*Dy      0         0      0;
             r*A      f        -g*A*Dx   -r      0;
            -f        r*A      -g*A*Dy    0     -r ]
end

# The three physical modes are the eigenvectors with the largest C-grid component
function physical_damping_rate(k, l; kw...)
    F = eigen(cd_symbol(k, l; kw...))
    c_grid_weight = [norm(F.vectors[1:3, n]) / norm(F.vectors[:, n]) for n in 1:5]
    physical = sortperm(c_grid_weight, rev=true)[1:3]
    return -maximum(real, F.values[physical])
end

Δ = 1e5
@info "Damping of the physical modes with relaxation_time = 10 days (Δ = 100 km, f = 1e-4 s⁻¹, c = 2.5 m s⁻¹)"
for n in (2, 3, 4, 6, 10, 20)
    κ = 2π / (n * Δ)
    rate = physical_damping_rate(κ / √2, κ / √2)
    unrelaxed = physical_damping_rate(κ / √2, κ / √2; relaxation_time=Inf)
    @info @sprintf("%2dΔx: e-folding %9.1f days, ν = %7.1f m² s⁻¹, ν₄ = %.2e m⁴ s⁻¹ (τ = ∞: rate %.1e s⁻¹)",
                   n, 1 / rate / day, rate / κ^2, rate / κ^4, unrelaxed)
end
