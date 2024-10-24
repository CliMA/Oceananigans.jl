Δt = 1200 # seconds
Aᵥ = 1e-2 # m²/s
SLV = 0.02

Δz = sqrt(4 * Aᵥ * Δt / SLV)




Nz = 15
Lz = 1800
σ = 1.3125
z_faces_2(k) = -Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
z_faces_2(Nz+1) - z_faces_2(Nz)