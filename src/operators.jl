# Functions to calculate the x, y, and z-derivatives on an Arakawa C-grid at
# every grid point:
#     δˣ(f) = (f)ᴱ - (f)ᵂ,   δʸ(f) = (f)ᴺ - (f)ˢ,   δᶻ(f) = (f)ᵀ - (f)ᴮ
# where the E, W, N, and S superscripts indicate that the value of f is
# evaluated on the eastern, western, northern, and southern walls of the cell,
# respectively. Similarly, the T and B superscripts indicate the top and bottom
# walls of the cell.
δˣ(f) = (f .- circshift(f, (1, 0, 0)))
δʸ(f) = (f .- circshift(f, (0, 1, 0)))
# δᶻ(f) = (circshift(f, (0, 0, -1)) - circshift(f, (0, 0, 1)))

function δᶻ(f)
  ff = Array{Float64, 3}(undef, size(f)...)

  ff[:, :, 1] = f[:, :, 2] - f[:, :, 1]          # δᶻ at top layer.
  ff[:, :, end] = f[:, :, end] - f[:, :, end-1]  # δᶻ at bottom layer.

  # δᶻ in the interior.
  ff[:, :, 2:end-1] = (f .- circshift(f, (0, 0, 1)))[:, :, 2:end-1]

  return ff
end

#=
Example function to compute an x-derivative:

function xderiv!(ux, u, grid)
  @views @. ux[2:grid.nx, :, :] = ( u[2:grid.nx, :, :] - u[1:grid.nx-1, :, :] ) / grid.dx
  @views @. ux[1,         :, :] = ( u[1,         :, :] - u[grid.nx,     :, :] ) / grid.dx
  nothing
end

However --- won't we need to know whether u lives in the cell center or cell face?
=#

# Functions to calculate the value of a quantity on a face as the average of
# the quantity in the two cells to which the face is common:
#     ̅qˣ = (qᴱ + qᵂ) / 2,   ̅qʸ = (qᴺ + qˢ) / 2,   ̅qᶻ = (qᵀ + qᴮ) / 2
# where the superscripts are as defined for the derivative operators.
avgˣ(f) = (f .+ circshift(f, (1, 0, 0))) / 2
avgʸ(f) = (f .+ circshift(f, (0, 1, 0))) / 2
# avgᶻ(f) = (circshift(f, (0, 0, -1)) + circshift(f, (0, 0, 1))) / 2

function avgᶻ(f)
  ff = Array{Float64, 3}(undef, size(Tⁿ)...)

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

#=
function xderiv!(out, in, g::Grid)
end

function xderiv(in, g)
  out = zero(in)
end
=#
# avgˣ(f) = @views (f + cat(f[2:end, :, :], f[1:1, :, :]; dims=1)) / 2
# avgʸ(f) = @views (f + cat(f[:, 2:end, :], f[:, 1:1, :]; dims=2)) / 2
# avgᶻ(f) = @views (f + cat(f[:, :, 2:end], f[:, :, 1:1]; dims=3)) / 2

# Calculate the divergence of the flux of a quantify f = (fˣ, fʸ, fᶻ) over the
# cell.
function div(fˣ, fʸ, fᶻ)
  Vᵘ = V
  (1/V) * ( δˣ(Aˣ .* fˣ) + δʸ(Aʸ .* fʸ) + δᶻ(Aᶻ .* fᶻ) )
end

# Calculate the divergence of a flux of Q over a zone with velocity field
# 𝐮 = (u,v,w): ∇ ⋅ (𝐮 Q).
function div_flux(u, v, w, Q)
  Vᵘ = V
  div_flux_x = δˣ(Aˣ .* u .* avgˣ(Q))
  div_flux_y = δʸ(Aʸ .* v .* avgʸ(Q))
  div_flux_z = δᶻ(Aᶻ .* w .* avgᶻ(Q))

  # Imposing zero vertical flux through the top and bottom layers.
  @. div_flux_z[:, :, 1] = 0
  @. div_flux_z[:, :, 50] = 0

  (1/Vᵘ) .* (div_flux_x .+ div_flux_y .+ div_flux_z)
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
  advection_x = δˣ(avgᶻ(Aˣ.*u) .* avgˣ(w))
  advection_y = δʸ(avgᶻ(Aʸ.*v) .* avgʸ(w))
  advection_z = δᶻ(avgᶻ(Aᶻ.*w) .* avgᶻ(w))
  (1/Vᵘ) .* (advection_x + advection_y + advection_z)
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

𝜈ʰ = 4e-2  # Horizontal eddy viscosity [Pa·s]. viscAh in MITgcm.
𝜈ᵛ = 4e-2  # Vertical eddy viscosity [Pa·s]. viscAz in MITgcm.

# Laplacian diffusion for face quantities: ∇ · (ν∇u)
function laplacian_diffusion_face(u)
  Vᵘ = V
  𝜈∇u_x = 𝜈ʰ .* avgˣ(Aˣ) .* δˣ(u)
  𝜈∇u_y = 𝜈ʰ .* avgʸ(Aʸ) .* δʸ(u)
  𝜈∇u_z = 𝜈ᵛ .* avgᶻ(Aᶻ) .* δᶻ(u)
  (1/Vᵘ) .* div(𝜈∇u_x, 𝜈∇u_y, 𝜈∇u_z)
end

horizontal_laplacian(f) = circshift(f, (1, 0, 0)) + circshift(f, (-1, 0, 0)) + circshift(f, (0, 1, 0)) + circshift(f, (0, -1, 0)) - 4 .* f
