# Inline helper functions.
@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

# Functions to calculate the x, y, and z-derivatives on an Arakawa C-grid at
# every grid point:
#     δˣ(f) = (f)ᴱ - (f)ᵂ,   δʸ(f) = (f)ᴺ - (f)ˢ,   δᶻ(f) = (f)ᵀ - (f)ᴮ
# where the E, W, N, and S superscripts indicate that the value of f is
# evaluated on the eastern, western, northern, and southern walls of the cell,
# respectively. Similarly, the T and B superscripts indicate the top and bottom
# walls of the cell.

#=
Some benchmarking with Nx, Ny, Nz = 200, 200, 200.

using BenchmarkTools

A = reshape(collect(0:Nx*Ny*Nz-1), (Nx, Ny, Nz));
B = zeros((Nx, Ny, Nz));

@btime δˣ($A);
  54.556 ms (22 allocations: 122.07 MiB)

@btime δˣb!($A, $B)  # With bounds checking.
  19.870 ms (0 allocations: 0 bytes)

@btime δˣ!($A, $B)  # With @inbounds. Looping in fast k, j, i order.
  16.862 ms (0 allocations: 0 bytes)

@btime δˣ!!($A, $B)  # With @inbounds. Looping in slow i, j, k order.
  92.987 ms (0 allocations: 0 bytes)
=#

# δˣc2f, δʸc2f, and δᶻc2f calculate a difference in the x, y, and
# z-directions for a field defined at the cell centers
# and projects it onto the cell faces.

# Input: Field defined at the u-faces, which has size (Nx, Ny, Nz).
# Output: Field defined at the cell centers, which has size (Nx, Ny, Nz).
function δˣc2f(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[i, j, k] - f[decmod1(i,Nx), j, k]
    end
    δf
end

# Input: Field defined at the v-faces, which has size (Nx, Ny, Nz).
# Output: Field defined at the cell centers, which has size (Nx, Ny, Nz).
function δʸc2f(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[i, j, k] - f[i, decmod1(j,Ny), k]
    end
    δf
end

# Input: Field defined at the w-faces, which has size (Nx, Ny, Nz).
# Output: Field defined at the cell centers, which has size (Nx, Ny, Nz).
function δᶻc2f(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 2:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[i, j, k] - f[i, j, decmod1(k,Nz)]
    end
    @. δf[:, :, 1] = 0
    δf
end

# δˣf2c, δʸf2c, and δᶻf2c calculate a difference in the x, y, and
# z-directions for a field defined at the cell faces
# and projects it onto the cell centers.

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the u-faces, which has size (Nx, Ny, Nz).
function δˣf2c(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[incmod1(i, Nx), j, k] - f[i, j, k]
    end
    δf
end

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the v-faces, which has size (Nx, Ny, Nz).
function δʸf2c(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[i, incmod1(j, Ny), k] - f[i, j, k]
    end
    δf
end

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the v-faces, which has size (Nx, Ny, Nz).
function δᶻf2c(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:(Nz-1), j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[i, j, incmod1(k, Nz)] - f[i, j, k]
    end
    @. δf[:, :, end] = 0
    δf
end

# function δˣ!(g::Grid, f, δˣf)
#     for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
#       @inbounds δˣf[i, j, k] = f[i, j, k] - f[decmod1(i, Nx), j, k]
#     end
# end
#
# function δʸ!(g::Grid, f, δʸf)
#     for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
#       @inbounds δˣf[i, j, k] = f[i, j, k] - f[decmod1(i, Nx), j, k]
#     end
# end

# Functions to calculate the value of a quantity on a face as the average of
# the quantity in the two cells to which the face is common:
#     ̅qˣ = (qᴱ + qᵂ) / 2,   ̅qʸ = (qᴺ + qˢ) / 2,   ̅qᶻ = (qᵀ + qᴮ) / 2
# where the superscripts are as defined for the derivative operators.

# In case avgⁱ is called on a scalar s, e.g. Aˣ on a RegularCartesianGrid, just
# return the scalar.
avgˣc2f(s::Number) = s
avgʸc2f(s::Number) = s
avgᶻc2f(s::Number) = s
avgˣf2c(s::Number) = s
avgʸf2c(s::Number) = s
avgᶻf2c(s::Number) = s

# Input: Field defined at the u-faces, which has size (Nx, Ny, Nz).
# Output: Field defined at the cell centers, which has size (Nx, Ny, Nz).
function avgˣc2f(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[i, j, k] + f[decmod1(i,Nx), j, k]) / 2
    end
    δf
end

# Input: Field defined at the v-faces, which has size (Nx, Ny, Nz).
# Output: Field defined at the cell centers, which has size (Nx, Ny, Nz).
function avgʸc2f(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[i, j, k] + f[i, decmod1(j,Ny), k]) / 2
    end
    δf
end

# Input: Field defined at the w-faces, which has size (Nx, Ny, Nz).
# Output: Field defined at the cell centers, which has size (Nx, Ny, Nz).
function avgᶻc2f(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 2:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[i, j, k] + f[i, j, decmod1(k,Nz)]) / 2
    end
    @. δf[:, :, 1] = 0
    δf
end

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the u-faces, which has size (Nx, Ny, Nz).
function avgˣf2c(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[incmod1(i, Nx), j, k] + f[i, j, k]) / 2
    end
    δf
end

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the v-faces, which has size (Nx, Ny, Nz).
function avgʸf2c(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[i, incmod1(j, Ny), k] + f[i, j, k]) / 2
    end
    δf
end

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the w-faces, which has size (Nx, Ny, Nz).
function avgᶻf2c(f)
    Nx, Ny, Nz = size(f)
    δf = zeros(Nx, Ny, Nz)
    for k in 1:(Nz-1), j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[i, j, incmod1(k, Nz)] + f[i, j, k]) / 2
    end
    @. δf[:, :, end] = 0
    δf
end

# Input: fˣ is on a u-face grid with size (Nx, Ny, Nz).
#        fʸ is on a v-face grid with size (Nx, Ny, Nz).
#        fᶻ is on a w-face grid with size (Nx, Ny, Nz).
# Output: ∇·̲f is on a zone/cell center grid with size (Nx, Ny, Nz).
function div_f2c(fˣ, fʸ, fᶻ)
    Vᵘ = V
    (1/Vᵘ) * ( δˣf2c(Aˣ .* fˣ) + δʸf2c(Aʸ .* fʸ) + δᶻf2c(Aᶻ .* fᶻ) )
end

# Input: fˣ is on a cell center grid with size (Nx, Ny, Nz).
#        fʸ is on a cell center grid with size (Nx, Ny, Nz).
#        fᶻ is on a cell center grid with size (Nx, Ny, Nz).
# Output: ∇·̲f is on a face grid with size (Nx, Ny, Nz). The exact face depends
#         on the quantitify f̃ = (fx, fy, fz) being differentiated.
function div_c2f(fˣ, fʸ, fᶻ)
    Vᵘ = V
    (1/Vᵘ) * ( δˣc2f(Aˣ .* fˣ) + δʸc2f(Aʸ .* fʸ) + δᶻc2f(Aᶻ .* fᶻ) )
end

# Calculate the divergence of a flux of Q over a zone with velocity field
# ũ = (u,v,w): ∇ ⋅ (ũQ).
# Input: u is on a u-face grid with size (Nx, Ny, Nz).
#        v is on a v-face grid with size (Nx, Ny, Nz).
#        w is on a w-face grid with size (Nx, Ny, Nz).
#        Q is on a zone/cell center grid with size (Nx, Ny, Nz).
# Output: ∇·(u̲Q) is on zone/cell center grid with size (Nx, Ny, Nz).
function div_flux_f2c(u, v, w, Q)
    Vᵘ = V
    flux_x = Aˣ .* u .* avgˣc2f(Q)
    flux_y = Aʸ .* v .* avgʸc2f(Q)
    flux_z = Aᶻ .* w .* avgᶻc2f(Q)

    # Imposing zero vertical flux through the top and bottom layers.
    @. flux_z[:, :, 1] = 0
    @. flux_z[:, :, end] = 0

    (1/Vᵘ) .* (δˣf2c(flux_x) .+ δʸf2c(flux_y) .+ δᶻf2c(flux_z))
end

# Calculate the nonlinear advection (inertiaL acceleration or convective
# acceleration in other fields) terms ∇ ⋅ (Vu), ∇ ⋅ (Vv), and ∇ ⋅ (Vw) where
# V = (u,v,w). Each component gets its own function for now until we can figure
# out how to combine them all into one function.
function ũ∇u(u, v, w)
  Vᵘ = V
  (1/Vᵘ) .* (δˣc2f(avgˣf2c(Aˣ.*u) .* avgˣf2c(u)) + δʸc2f(avgˣf2c(Aʸ.*v) .* avgʸf2c(u)) + δᶻc2f(avgˣf2c(Aᶻ.*w) .* avgᶻf2c(u)))
end

function ũ∇v(u, v, w)
  Vᵘ = V
  (1/Vᵘ) .* (δˣc2f(avgʸf2c(Aˣ.*u) .* avgˣf2c(v)) + δʸc2f(avgʸf2c(Aʸ.*v) .* avgʸf2c(v)) + δᶻc2f(avgʸf2c(Aᶻ.*w) .* avgᶻf2c(v)))
end

function ũ∇w(u, v, w)
  Vᵘ = V
  uŵ_transport = avgᶻf2c(Aˣ.*u) .* avgˣf2c(w)
  vŵ_transport = avgᶻf2c(Aʸ.*v) .* avgʸf2c(w)
  wŵ_transport = avgᶻf2c(Aᶻ.*w) .* avgᶻf2c(w)

  wŵ_transport[:, :, 1]  .= 0
  wŵ_transport[:, :, end] .= 0

  (1/Vᵘ) .* (δˣc2f(uŵ_transport) .+ δʸc2f(vŵ_transport) .+ δᶻc2f(wŵ_transport))
end

κʰ = 4e-2  # Horizontal Laplacian heat diffusion [m²/s]. diffKhT in MITgcm.
κᵛ = 4e-2  # Vertical Laplacian heat diffusion [m²/s]. diffKzT in MITgcm.

# Laplacian diffusion for zone quantities: ∇ · (κ∇Q)
# Input: Q is on a cell centered grid with size (Nx, Ny, Nz).
# Output: ∇·(κ∇Q) is on a cell centered grid with size (Nx, Ny, Nz).
function κ∇²(Q)
  Vᵘ = V
  κ∇Q_x = κʰ .* Aˣ .* δˣc2f(Q)
  κ∇Q_y = κʰ .* Aʸ .* δʸc2f(Q)
  κ∇Q_z = κᵛ .* Aᶻ .* δᶻc2f(Q)
  (1/Vᵘ) .* div_f2c(κ∇Q_x, κ∇Q_y, κ∇Q_z)
end

𝜈ʰ = 4e-2  # Horizontal eddy viscosity [Pa·s]. viscAh in MITgcm.
𝜈ᵛ = 4e-2  # Vertical eddy viscosity [Pa·s]. viscAz in MITgcm.

# Laplacian diffusion for horizontal face quantities: ∇ · (ν∇u)
function 𝜈ʰ∇²(u)
  Vᵘ = V
  𝜈∇u_x = 𝜈ʰ .* avgˣf2c(Aˣ) .* δˣf2c(u)
  𝜈∇u_y = 𝜈ʰ .* avgʸf2c(Aʸ) .* δʸf2c(u)
  𝜈∇u_z = 𝜈ᵛ .* avgᶻf2c(Aᶻ) .* δᶻf2c(u)
  (1/Vᵘ) .* div_c2f(𝜈∇u_x, 𝜈∇u_y, 𝜈∇u_z)
end

# Laplacian diffusion for vertical face quantities: ∇ · (ν∇w)
function 𝜈ᵛ∇²(u)
  Vᵘ = V
  𝜈∇u_x = 𝜈ʰ .* avgˣf2c(Aˣ) .* δˣf2c(u)
  𝜈∇u_y = 𝜈ʰ .* avgʸf2c(Aʸ) .* δʸf2c(u)
  𝜈∇u_z = 𝜈ᵛ .* avgᶻf2c(Aᶻ) .* δᶻf2c(u)

  # Imposing free slip viscous boundary conditions at the bottom layer.
  @. 𝜈∇u_z[:, :,  1] = 0
  @. 𝜈∇u_z[:, :, end] = 0

  (1/Vᵘ) .* div_c2f(𝜈∇u_x, 𝜈∇u_y, 𝜈∇u_z)
end

horizontal_laplacian(f) = circshift(f, (1, 0, 0)) + circshift(f, (-1, 0, 0)) + circshift(f, (0, 1, 0)) + circshift(f, (0, -1, 0)) - 4 .* f

laplacian(f) = circshift(f, (1, 0, 0)) + circshift(f, (-1, 0, 0)) + circshift(f, (0, 1, 0)) + circshift(f, (0, -1, 0)) + circshift(f, (0, 0, 1)) + circshift(f, (0, -1, 0)) - 6 .* f

function laplacian3d_ppn(f)
    Nx, Ny, Nz = size(f)
    ∇²f = zeros(Nx, Ny, Nz)
    for k in 2:(Nz-1), j in 1:Ny, i in 1:Nx
       ∇²f[i, j, k] = f[incmod1(i, Nx), j, k] + f[decmod1(i, Nx), j, k] + f[i, incmod1(j, Ny), k] + f[i, decmod1(j, Ny), k] + f[i, j, k+1] + f[i, j, k-1] - 6*f[i, j, k]
    end
    for j in 1:Ny, i in 1:Nx
        ∇²f[i, j,   1] = -(f[i, j,     1] - f[i, j,   2]) + f[incmod1(i, Nx), j,   1] + f[decmod1(i, Nx), j,   1] + f[i, incmod1(j, Ny),   1] + f[i, decmod1(j, Ny),   1] - 4*f[i, j,   1]
        ∇²f[i, j, end] =  (f[i, j, end-1] - f[i, j, end]) + f[incmod1(i, Nx), j, end] + f[decmod1(i, Nx), j, end] + f[i, incmod1(j, Ny), end] + f[i, decmod1(j, Ny), end] - 4*f[i, j, end]
    end
    ∇²f
end
