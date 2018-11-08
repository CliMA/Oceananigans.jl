# Functions to calculate the x, y, and z-derivatives on an Arakawa C-grid at
# every grid point:
#     δˣ(f) = (f)ᴱ - (f)ᵂ,   δʸ(f) = (f)ᴺ - (f)ˢ,   δᶻ(f) = (f)ᵀ - (f)ᴮ
# where the E, W, N, and S superscripts indicate that the value of f is
# evaluated on the eastern, western, northern, and southern walls of the cell,
# respectively. Similarly, the T and B superscripts indicate the top and bottom
# walls of the cell.
δˣ(f::Array{NumType, 3}) = (f - cat(f[2:end,:,:], f[1:1,:,:]; dims=1)) / Δx
δʸ(f::Array{NumType, 3}) = (f - cat(f[:,2:end,:], f[:,1:1,:]; dims=2)) / Δy
δᶻ(f::Array{NumType, 3}) = (f - cat(f[:,:,2:end], f[:,:,1:1]; dims=3)) / Δz

function xderiv!(ux, u, grid)
  @views @. ux[2:grid.nx, :, :] = ( u[2:grid.nx, :, :] - u[1:grid.nx-1, :, :] ) / grid.dx
  @views @. ux[1,         :, :] = ( u[1,         :, :] - u[grid.nx,     :, :] ) / grid.dx 
  nothing
end

function xderivplus!(ux, u, grid)
  @views @. ux[1:grid.nx-1, :, :] = ( u[2:grid.nx, :, :] - u[1:grid.nx-1, :, :] ) / grid.dx
  @views @. ux[grid.nx,     :, :] = ( u[1,         :, :] - u[grid.nx,     :, :] ) / grid.dx 
  nothing
end

function yderiv!(uy, u, grid)
  @views @. uy[:, 2:grid.ny, :] = ( u[:, 2:grid.ny, :] - u[:, 1:grid.ny-1, :] ) / grid.dy
  @views @. uy[:, 1,         :] = ( u[:, 1,         :] - u[:, grid.ny,     :] ) / grid.dy 
  nothing
end

function zderiv!(uz, u, grid)
  @views @. uz[:, :, 2:grid.nz] = ( u[:, :, 2:grid.nz] - u[:, :, 1:grid.nz-1] ) / grid.dz
  @views @. uz[:, :, 1        ] = ( u[:, :, 1        ] - u[:, :, grid.nz    ] ) / grid.dz 
  nothing
end

# Functions to calculate the value of a quantity on a face as the average of
# the quantity in the two cells to which the face is common:
#     ̅qˣ = (qᴱ + qᵂ) / 2,   ̅qʸ = (qᴺ + qˢ) / 2,   ̅qᶻ = (qᵀ + qᴮ) / 2
# where the superscripts are as defined for the derivative operators.
avgˣ(f::Array{NumType, 3}) = (f + cat(f[2:end,:,:], f[1:1,:,:]; dims=1)) / 2
avgʸ(f::Array{NumType, 3}) = (f + cat(f[:,2:end,:], f[:,1:1,:]; dims=2)) / 2
avgᶻ(f::Array{NumType, 3}) = (f + cat(f[:,:,2:end], f[:,:,1:1]; dims=3)) / 2

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

# Calculate the divergence of a flux of Q with velocity field V = (u,v,w):
# ∇ ⋅ (VQ).
function div_flux(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3}, Q::Array{NumType, 3})
  Vᵘ = V
  div_flux_x = δˣ(Aˣ .* u .* avgˣ(Q))
  div_flux_y = δʸ(Aʸ .* v .* avgʸ(Q))
  div_flux_z = δᶻ(Aᶻ .* w .* avgᶻ(Q))
  return (1/Vᵘ) .* (div_flux_x .+ div_flux_y .+ div_flux_z)
end

# Calculate the nonlinear advection (inertiaL acceleration or convective
# acceleration in other fields) terms ∇ ⋅ (Vu), ∇ ⋅ (Vv), and ∇ ⋅ (Vw) where
# V = (u,v,w). Each component gets its own function for now until we can figure
# out how to combine them all into one function.
function u_dot_u(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3})
  Vᵘ = V
  advection_x = δˣ(avgˣ(Aˣ.*u) .* avgˣ(u))
  advection_y = δʸ(avgˣ(Aʸ.*v) .* avgʸ(u))
  advection_z = δᶻ(avgˣ(Aᶻ.*w) .* avgᶻ(u))
  return (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end

function u_dot_v(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3})
  Vᵘ = V
  advection_x = δˣ(avgʸ(Aˣ.*u) .* avgˣ(v))
  advection_y = δʸ(avgʸ(Aʸ.*v) .* avgʸ(v))
  advection_z = δᶻ(avgʸ(Aᶻ.*w) .* avgᶻ(v))
  return (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end

function u_dot_w(u::Array{NumType, 3}, v::Array{NumType, 3},
  w::Array{NumType, 3})
  Vᵘ = V
  advection_x = δˣ(avgᶻ(Aˣ.*u) .* avgˣ(w))
  advection_y = δʸ(avgᶻ(Aʸ.*v) .* avgʸ(w))
  advection_z = δᶻ(avgᶻ(Aᶻ.*w) .* avgᶻ(w))
  return (1/Vᵘ) .* (advection_x + advection_y + advection_z)
end
