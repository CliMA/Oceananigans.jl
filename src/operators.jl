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
avgˣ(f) = (f .+ circshift(f, (1, 0, 0))) / 2
avgʸ(f) = (f .+ circshift(f, (0, 1, 0))) / 2
# avgᶻ(f) = (circshift(f, (0, 0, -1)) + circshift(f, (0, 0, 1))) / 2

function avgᶻ(f)
  ff = Array{Float64, 3}(undef, size(f)...)

  ff[:, :, 1] = (f[:, :, 2] + f[:, :, 1]) / 2          # avgᶻ at top layer.
  ff[:, :, end] = (f[:, :, end] + f[:, :, end-1]) / 2  # avgᶻ at bottom layer.

  # avgᶻ in the interior.
  ff[:, :, 2:end-1] = (f .+ circshift(f, (0, 0, 1)))[:, :, 2:end-1] ./ 2

  return ff
end

# In case avgⁱ is called on a scalar s, e.g. Aˣ on a RegularCartesianGrid, just
# return the scalar.
avgˣ(s::Number) = s
avgʸ(s::Number) = s
avgᶻ(s::Number) = s

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the u-faces, which has size (Nx+1, Ny, Nz).
function avgˣz2f(f)
    Nx, Ny, Nz = size(f)
    fa = zeros(Nx+1, Ny, Nz)

    # Calculate avgˣ in the interior.
    for k in 1:Nz, j in 1:Ny, i in 2:Nx
        fa[i, j, k] =  (f[i-1, j, k] + f[i, j, k]) / 2
    end

    # Calculate avgˣ at the left and right boundaries (the leftmost and rightmost faces are the
    # same in our periodic configuration).
    for k in 1:Nz, j in 1:Ny
        avg′ = (f[1, j, k] + f[end, j, k]) / 2
        fa[1, j, k], fa[end, j, k] = avg′, avg′
    end

    fa
end

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the v-faces, which has size (Nx, Ny+1, Nz).
function avgʸz2f(f)
    Nx, Ny, Nz = size(f)
    fa = zeros(Nx, Ny+1, Nz)

    # Calculate avgʸ in the interior.
    for k in 1:Nz, j in 2:Ny, i in 1:Nx
        fa[i, j, k] =  (f[i, j-1, k] + f[i, j, k]) / 2
    end

    # Calculate avgʸ at the north and south boundaries (the northmost and southtmost faces are the
    # same in our periodic configuration).
    for k in 1:Nz, i in 1:Nx
        avg′ = (f[i, 1, k] + f[i, end, k]) / 2
        fa[i, 1, k], fa[i, end, k] = avg′, avg′
    end

    fa
end

# Input: Field defined at the cell centers, which has size (Nx, Ny, Nz).
# Output: Field defined at the w-faces, which has size (Nx, Ny, Nz+1).
function avgᶻz2f(f)
    Nx, Ny, Nz = size(f)
    fa = zeros(Nx, Ny, Nz+1)

    # Calculate avgᶻ in the interior.
    for k in 2:Nz, j in 1:Ny, i in 1:Nx
        fa[i, j, k] =  (f[i, j, k-1] + f[i, j, k]) / 2
    end

    # Calculate avgᶻ at the top and bottom boundaries (the surface and bottom faces are the
    # same in our periodic configuration).
    for j in 1:Ny, i in 1:Nx
        avg′ = (f[i, j, 1] + f[i, j, end]) / 2
        fa[i, j, 1], fa[i, j, end] = avg′, avg′
    end

    fa
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
function div_f2z(fˣ, fʸ, fᶻ)
    Vᵘ = V
    (1/Vᵘ) * ( δˣf2z(Aˣ .* fˣ) + δʸf2z(Aʸ .* fʸ) + δᶻf2z(Aᶻ .* fᶻ) )
end

# Calculate the divergence of a flux of Q over a zone with velocity field
# 𝐮 = (u,v,w): ∇ ⋅ (𝐮 Q).
function div_flux(u, v, w, Q)
  Vᵘ = V
  flux_x = Aˣ .* u .* avgˣ(Q)
  flux_y = Aʸ .* v .* avgʸ(Q)
  flux_z = Aᶻ .* w .* avgᶻ(Q)

  # Imposing zero vertical flux through the top and bottom layers.
  @. flux_z[:, :, 1] = 0
  @. flux_z[:, :, end] = 0

  (1/Vᵘ) .* (δˣ(flux_x) .+ δʸ(flux_y) .+ δᶻ(flux_z))
end

# Input: u is on a u-face grid with size (Nx+1, Ny, Nz).
#        v is on a v-face grid with size (Nx, Ny+1, Nz).
#        w is on a w-face grid with size (Nx, Ny, Nz+1).
#        Q is on a zone/cell center grid with size (Nx, Ny, Nz).
# Output: ∇·(u̲Q) is on zone/cell center grid with size (Nx, Ny, Nz).
function div_flux_f2z(u, v, w, Q)
    Vᵘ = V
    flux_x = Aˣ .* u .* avgˣz2f(Q)
    flux_y = Aʸ .* v .* avgʸz2f(Q)
    flux_z = Aᶻ .* w .* avgᶻz2f(Q)

    # Imposing zero vertical flux through the top and bottom layers.
    @. flux_z[:, :, 1] = 0
    @. flux_z[:, :, end] = 0

    (1/Vᵘ) .* (δˣf2z(flux_x) .+ δʸf2z(flux_y) .+ δᶻf2z(flux_z))
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
