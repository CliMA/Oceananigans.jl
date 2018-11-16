using Printf

import FFTW

Nˣ, Nʸ, Nᶻ = 100, 100, 50  # Number of grid points in (x,y,z).
Lˣ, Lʸ, Lᶻ = 2000, 2000, 1000  # Domain size [m].

Δx, Δy, Δz = Lˣ/Nˣ, Lʸ/Nʸ, Lᶻ/Nᶻ  # Grid spacing [m].
Aˣ, Aʸ, Aᶻ = Δy*Δz, Δx*Δz, Δx*Δy  # Cell face areas [m²].
V = Δx*Δy*Δz  # Volume of a cell [m³].

# Precalculate wavenumbers and prefactor required to convert Fourier
# coefficients during the spectral pressure solve.
kˣ² = (2 / Δx^2) .* (cos.(π*(0:Nˣ-1) / Nˣ) .- 1)
kʸ² = (2 / Δy^2) .* (cos.(π*(0:Nʸ-1) / Nʸ) .- 1)
kᶻ² = (2 / Δz^2) .* (cos.(π*(0:Nᶻ-1) / Nᶻ) .- 1)
# kˣ² = (2 / Δx^2) .* (sin.(π*(1:Nˣ) / Nˣ) .- 1)
# kʸ² = (2 / Δy^2) .* (sin.(π*(1:Nʸ) / Nʸ) .- 1)
# kᶻ² = (2 / Δz^2) .* (sin.(π*(1:Nᶻ) / Nᶻ) .- 1)
# kˣ² = - (2 .* sin.(π*(1:Nˣ) / Nˣ) / Δx).^2
# kʸ² = - (2 .* sin.(π*(1:Nʸ) / Nʸ) / Δy).^2
# kᶻ² = - (2 .* sin.(π*(1:Nᶻ) / Nᶻ) / Δz).^2

kˣ² = repeat(reshape(kˣ², Nˣ, 1, 1), 1, Nʸ, Nᶻ)
kʸ² = repeat(reshape(kʸ², 1, Nʸ, 1), Nˣ, 1, Nᶻ)
kᶻ² = repeat(reshape(kᶻ², 1, 1, Nᶻ), Nˣ, Nʸ, 1)

k̃⁻² = 1 ./ (kˣ² .+ kʸ² .+ kᶻ²)
# @. k̃⁻² = - 1 / (kˣ² + kʸ² + kᶻ²)

k̃⁻²[1, 1, 1] = 0  # Solvability condition: DC component is zero.

# Solve an elliptic Poisson equation ∇²ϕ = ℱ for the pressure field ϕ where
# ℱ = ∇ ⋅ G and G = (Gᵘ,Gᵛ,Gʷ) using the Fourier-spectral method.
function solve_for_pressure(Gᵘ, Gᵛ, Gʷ)
  f = div(Gᵘ, Gᵛ, Gʷ)  # Right hand side or source term.

  # RHS_hat, fft_t, fft_bytes, fft_gc = @timed FFTW.fft(RHS)
  # RHS_hat, fft_t, fft_bytes, fft_gc = @timed FFTW.fft(FFTW.r2r(RHS, FFTW.REDFT10, 3), [1, 2])
  # RHS_hat, fft_t, fft_bytes, fft_gc = @timed FFTW.r2r(FFTW.fft(RHS, [1, 2]), FFTW.REDFT10, 3)
  # RHS_hat, fft_t, fft_bytes, fft_gc = @timed FFTW.r2r(RHS, FFTW.REDFT10)
  # f̂, fft_t, fft_bytes, fft_gc = @timed FFTW.rfft(FFTW.dct(f, 3), [1, 2])
  f̂, fft_t, fft_bytes, fft_gc = @timed FFTW.fft(FFTW.dct(f, 3), [1, 2])

  # TODO: How to do this using rfft as f̂ changes size to (Nˣ/2 + 1, Nʸ, Nᶻ)?
  p̂, hat_t, hat_bytes, hat_gc = @timed k̃⁻² .* f̂

  p, ifft_t, ifft_bytes, ifft_gc = @timed FFTW.idct(FFTW.ifft(p̂, [1, 2]), 3)
  # p, ifft_t, ifft_bytes, ifft_gc = @timed FFTW.idct(FFTW.rfft(p̂, Nˣ, [1, 2]), 3)
  # φ, ifft_t, ifft_bytes, ifft_gc = @timed FFTW.ifft(φ_hat)
  # φ, ifft_t, ifft_bytes, ifft_gc = @timed FFTW.r2r(FFTW.ifft(φ_hat, [1, 2]), FFTW.REDFT01, 3)
  # φ, ifft_t, ifft_bytes, ifft_gc = @timed FFTW.fft(FFTW.r2r(φ_hat, FFTW.REDFT01, 3), [1, 2]) / (2 * Nˣ * Nʸ * Nᶻ)
  # φ, ifft_t, ifft_bytes, ifft_gc = @timed FFTW.r2r(φ_hat, FFTW.REDFT01) ./ (Nˣ * Nʸ * Nᶻ)

  @info begin
    string("Fourier-spectral profiling:\n",
           @sprintf("fFFT: (%.0f ms, %.2f MiB, %.1f%% GC)\n", fft_t*1000, fft_bytes/1024^2, fft_gc*100),
           @sprintf("FFTc: (%.0f ms, %.2f MiB, %.1f%% GC)\n", hat_t*1000, hat_bytes/1024^2, hat_gc*100),
           @sprintf("iFFT: (%.0f ms, %.2f MiB, %.1f%% GC)\n", ifft_t*1000, ifft_bytes/1024^2, ifft_gc*100))
  end
  @info "p[50, 50, 1] = $(p[50, 50, 1])"
  return real.(p)
end
