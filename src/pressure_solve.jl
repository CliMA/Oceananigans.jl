import FFTW

# Precalculate wavenumbers and prefactor required to convert Fourier
# coefficients during the spectral pressure solve.
kˣ² = (2 / Δx^2) .* (cos.(π*(1:Nˣ) / Nˣ) .- 1)
kʸ² = (2 / Δy^2) .* (cos.(π*(1:Nʸ) / Nʸ) .- 1)
kᶻ² = (2 / Δz^2) .* (cos.(π*(1:Nᶻ) / Nᶻ) .- 1)

kˣ² = repeat(reshape(kˣ², Nˣ, 1, 1), 1, Nʸ, Nᶻ)
kʸ² = repeat(reshape(kʸ², 1, Nʸ, 1), Nˣ, 1, Nᶻ)
kᶻ² = repeat(reshape(kᶻ², 1, 1, Nᶻ), Nˣ, Nʸ, 1)

prefactor = 1 ./ (kˣ² .+ kʸ² .+ kᶻ²)
prefactor[1,1,1] = 0  # Solvability condition: DC component is zero.

# Solve an elliptic Poisson equation ∇²ϕ = ℱ for the pressure field ϕ where
# ℱ = ∇ ⋅ G and G = (Gᵘ,Gᵛ,Gʷ) using the Fourier-spectral method.
function solve_for_pressure(Gᵘ, Gᵛ, Gʷ)
  RHS = δˣ(Gᵘ) + δʸ(Gᵛ) + δᶻ(Gʷ)
  RHS_hat = FFTW.fft(RHS)
  φ_hat = prefactor .* RHS_hat
  φ = FFTW.ifft(φ_hat)
  return real.(φ)
end
