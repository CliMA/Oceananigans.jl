using Oceananigans

# Definition of the "Bickley jet": a sech(y)^2 jet with sinusoidal tracer
Ψ(y) = - tanh(y)
U(y) = sech(y)^2

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

"""
    u, v: Large-scale jet + vortical perturbations
       c: Sinusoid
"""
function set_bickley_jet!(model;
                          ϵ = 0.1, # perturbation magnitude
                          ℓ = 0.5, # gaussian width
                          k = 0.5) # sinusoidal wavenumber
    
    # total initial conditions
    uᵢ(x, y, z) = U(y) + ϵ * ũ(x, y, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(x, y, ℓ, k)
    cᵢ(x, y, z) = C(y, model.grid.Ly)
    
    # Note that u, v are only horizontally-divergence-free as resolution -> ∞.
    set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

    return nothing
end

bickley_grid(; Nh, arch=CPU(), halo=(3, 3, 3)) = RectilinearGrid(arch; size=(Nh, Nh, 1), halo,
                                                                 x = (-2π, 2π), y=(-2π, 2π), z=(0, 1),
                                                                 topology = (Periodic, Periodic, Bounded))

