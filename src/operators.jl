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
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[i, j, k] - f[i, j, decmod1(k,Nz)]
    end
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
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  f[i, j, incmod1(k, Nz)] - f[i, j, k]
    end
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
avgˣ(s::Number) = s
avgʸ(s::Number) = s
avgᶻ(s::Number) = s

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
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[i, j, k] + f[i, j, decmod1(k,Nz)]) / 2
    end
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
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        δf[i, j, k] =  (f[i, j, incmod1(k, Nz)] + f[i, j, k]) / 2
    end
    δf
end

# Calculate the divergence of the flux of a quantify f = (fˣ, fʸ, fᶻ) over the
# cell.
function div(fˣ, fʸ, fᶻ)
  Vᵘ = V
  (1/V) * ( δˣ(Aˣ .* fˣ) + δʸ(Aʸ .* fʸ) + δᶻ(Aᶻ .* fᶻ) )
end

# Input: fˣ is on a u-face grid with size (Nx+1, Ny, Nz).
#        fʸ is on a v-face grid with size (Nx, Ny+1, Nz).
#        fᶻ is on a w-face grid with size (Nx, Ny, Nz+1).
# Output: ∇·̲f is on a zone/cell center grid with size (Nx, Ny, Nz).
function div_f2c(fˣ, fʸ, fᶻ)
    Vᵘ = V
    (1/Vᵘ) * ( δˣf2c(Aˣ .* fˣ) + δʸf2c(Aʸ .* fʸ) + δᶻf2c(Aᶻ .* fᶻ) )
end

# # Input: fˣ is on a u-face grid with size (Nx, Ny, Nz).
# #        fʸ is on a v-face grid with size (Nx, Ny, Nz).
# #        fᶻ is on a w-face grid with size (Nx, Ny, Nz).
# # Output: ∇·̲f is on a zone/cell center grid with size (Nx, Ny, Nz).
# function div_c2f(fˣ, fʸ, fᶻ)
#     Vᵘ = V
#     (1/Vᵘ) * ( δˣc2f(Aˣ .* fˣ) + δʸc2f(Aʸ .* fʸ) + δᶻc2f(Aᶻ .* fᶻ) )
# end

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
function u_dot_u(u, v, w)
  Vᵘ = V
  advection_x = δˣ(avgˣ(Aˣ.*u) .* avgˣ(u))
  advection_y = δʸ(avgˣ(Aʸ.*v) .* avgʸ(u))
  advection_z = δᶻ(avgˣ(Aᶻ.*w) .* avgᶻ(u))
  (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end

function u_dot_v(u, v, w)
  Vᵘ = V
  advection_x = δˣ(avgʸ(Aˣ.*u) .* avgˣ(v))
  advection_y = δʸ(avgʸ(Aʸ.*v) .* avgʸ(v))
  advection_z = δᶻ(avgʸ(Aᶻ.*w) .* avgᶻ(v))
  (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end

function u_dot_w(u, v, w)
  Vᵘ = V
  uŵ_transport = avgᶻ(Aˣ.*u) .* avgˣ(w)
  vŵ_transport = avgᶻ(Aʸ.*v) .* avgʸ(w)
  wŵ_transport = avgᶻ(Aᶻ.*w) .* avgᶻ(w)

  wŵ_transport[:, :, 1]  .= 0
  wŵ_transport[:, :, 50] .= 0

  (1/Vᵘ) .* (δˣ(uŵ_transport) .+ δʸ(vŵ_transport) .+ δᶻ(wŵ_transport))
end

κʰ = 4e-2  # Horizontal Laplacian heat diffusion [m²/s]. diffKhT in MITgcm.
κᵛ = 4e-2  # Vertical Laplacian heat diffusion [m²/s]. diffKzT in MITgcm.

# Laplacian diffusion for zone quantities: ∇ · (κ∇Q)
function laplacian_diffusion_zone(Q)
  Vᵘ = V
  κ∇Q_x = κʰ .* Aˣ .* δˣ(Q)
  κ∇Q_y = κʰ .* Aʸ .* δʸ(Q)
  κ∇Q_z = κᵛ .* Aᶻ .* δᶻ(Q)
  (1/Vᵘ) .* div(κ∇Q_x, κ∇Q_y, κ∇Q_z)
end

# Input: Q is on a zone/cell centered grid with size (Nx, Ny, Nz).
# Output: ∇·(κ∇Q) is on a zone/cell centered grid with size (Nx, Ny, Nz).
function laplacian_diffusion_z2z(Q)
    Vᵘ = V
    κ∇Q_x = κʰ .* Aˣ .* δˣz2f(Q)
    κ∇Q_y = κʰ .* Aʸ .* δʸz2f(Q)
    κ∇Q_z = κᵛ .* Aᶻ .* δᶻz2f(Q)
    (1/Vᵘ) .* div_f2z(κ∇Q_x, κ∇Q_y, κ∇Q_z)
end

𝜈ʰ = 4e-2  # Horizontal eddy viscosity [Pa·s]. viscAh in MITgcm.
𝜈ᵛ = 4e-2  # Vertical eddy viscosity [Pa·s]. viscAz in MITgcm.

# Laplacian diffusion for horizontal face quantities: ∇ · (ν∇u)
function laplacian_diffusion_face_h(u)
  Vᵘ = V
  𝜈∇u_x = 𝜈ʰ .* avgˣ(Aˣ) .* δˣ(u)
  𝜈∇u_y = 𝜈ʰ .* avgʸ(Aʸ) .* δʸ(u)
  𝜈∇u_z = 𝜈ᵛ .* avgᶻ(Aᶻ) .* δᶻ(u)

  # Imposing free slip viscous boundary conditions at the bottom layer.
  # @. 𝜈∇u_x[:, :, 50] = 0
  # @. 𝜈∇u_y[:, :, 50] = 0

  (1/Vᵘ) .* div(𝜈∇u_x, 𝜈∇u_y, 𝜈∇u_z)
end

# Laplacian diffusion for vertical face quantities: ∇ · (ν∇w)
function laplacian_diffusion_face_v(u)
  Vᵘ = V
  𝜈∇u_x = 𝜈ʰ .* avgˣ(Aˣ) .* δˣ(u)
  𝜈∇u_y = 𝜈ʰ .* avgʸ(Aʸ) .* δʸ(u)
  𝜈∇u_z = 𝜈ᵛ .* avgᶻ(Aᶻ) .* δᶻ(u)

  # Imposing free slip viscous boundary conditions at the bottom layer.
  @. 𝜈∇u_z[:, :,  1] = 0
  @. 𝜈∇u_z[:, :, 50] = 0

  (1/Vᵘ) .* div(𝜈∇u_x, 𝜈∇u_y, 𝜈∇u_z)
end

horizontal_laplacian(f) = circshift(f, (1, 0, 0)) + circshift(f, (-1, 0, 0)) + circshift(f, (0, 1, 0)) + circshift(f, (0, -1, 0)) - 4 .* f
laplacian(f) = circshift(f, (1, 0, 0)) + circshift(f, (-1, 0, 0)) + circshift(f, (0, 1, 0)) + circshift(f, (0, -1, 0)) + circshift(f, (0, 0, 1)) + circshift(f, (0, -1, 0)) - 6 .* f
