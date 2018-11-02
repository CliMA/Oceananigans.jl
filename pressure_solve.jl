import FFTW

# Precalculate wavenumbers and prefactor required to convert Fourier
# coefficients during the spectral pressure solve.
kˣ² = (2 / Δx^2) .* (cos.(π*[1:Nˣ;] / Nˣ) .- 1)
kʸ² = (2 / Δy^2) .* (cos.(π*[1:Nʸ;] / Nʸ) .- 1)
kᶻ² = (2 / Δz^2) .* (cos.(π*[1:Nᶻ;] / Nᶻ) .- 1)

kˣ² = repeat(reshape(kˣ², Nˣ, 1, 1), 1, Nʸ, Nᶻ)
kʸ² = repeat(reshape(kʸ², 1, Nʸ, 1), Nˣ, 1, Nᶻ)
kᶻ² = repeat(reshape(kᶻ², 1, 1, Nᶻ), Nˣ, Nʸ, 1)

prefactor = 1 ./ (kˣ² .+ kʸ² .+ kᶻ²)
prefactor[1,1,1] = 0  # Solvability condition: DC component is zero.

# Solve an elliptic Poisson equation ∇²ϕ = ℱ for the pressure field ϕ where
# ℱ = ∇ ⋅ G and G = (Gᵘ,Gᵛ,Gʷ) using the Fourier-spectral method.
function solve_for_pressure(Gᵘ, Gᵛ, Gʷ)
  RHS = δˣ(Gᵘ) + δʸ(Gᵛ) + δᶻ(Gʷ)

  RHS_hat, fft_t, fft_bytes, fft_gc = @timed FFTW.fft(RHS)
  φ_hat, hat_t, hat_bytes, hat_gc = @timed prefactor .* RHS_hat
  φ, ifft_t, ifft_bytes, ifft_gc = @timed FFTW.ifft(φ_hat)

  @info begin
    string("Fourier-spectral profiling:\n",
           "FFT: ($fft_t s, $fft_bytes bytes, $fft_gc GC)\n",
           "FFT: ($hat_t s, $hat_bytes bytes, $hat_gc GC)\n",
           "FFT: ($ifft_t s, $ifft_bytes bytes, $ifft_gc GC)\n")
  end

  return real.(φ)
end
