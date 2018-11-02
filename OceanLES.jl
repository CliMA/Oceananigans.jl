# Importing packages and functions as needed.
import FFTW
using Statistics: mean

include("constants.jl")
include("operators.jl")
include("equation_of_state.jl")
include("pressure_solve.jl")
include("generate_initial_conditions.jl")

@info begin
  string("Ocean LES model parameters:\n",
         "NumType: $NumType\n",
         "(Nˣ, Nʸ, Nᶻ) = ($Nˣ, $Nʸ, $Nᶻ) [m]\n",
         "(Lˣ, Lʸ, Lᶻ) = ($Lˣ, $Lʸ, $Lᶻ) [m]\n",
         "(Δx, Δy, Δz) = ($Δx, $Δy, $Δz) [m]\n",
         "(Aˣ, Aʸ, Aᶻ) = ($Aˣ, $Aʸ, $Aᶻ) [m²]\n",
         "V = $V [m³]\n",
         "Nᵗ = $Nᵗ [s]\n",
         "Δt = $Δt [s]\n")
end

# Initialize arrays used to store source terms at current and previous
# timesteps, and other variables.
Gᵘⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵘⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

Gᵘⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵘⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ⁻¹ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

Gᵘⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵘⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gʷⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gᵀⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
Gˢⁿ⁺ʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)

pⁿʰ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)
for n in 1:Nᵗ
  # Calculate new density and density deviation.
  δρ .= ρ.(Tⁿ, Sⁿ, pⁿ) .- ρⁿ
  ρⁿ = ρⁿ + δρ

  # Store source terms from previous iteration.
  Gᵘⁿ⁻¹ = Gᵘⁿ; Gᵛⁿ⁻¹ = Gᵛⁿ; Gʷⁿ⁻¹ = Gʷⁿ; Gᵀⁿ⁻¹ = Gᵀⁿ; Gˢⁿ⁻¹ = Gˢⁿ;

  # Calculate source terms for the current time step.
  Gˢⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Sⁿ) + Fˢ
  Gᵀⁿ = -div_flux(uⁿ, vⁿ, wⁿ, Tⁿ) + Fᵀ

  Gᵘⁿ = -u_dot_u(uⁿ, vⁿ, wⁿ) + f.*vⁿ + Fᵘ
  Gᵛⁿ = -u_dot_v(uⁿ, vⁿ, wⁿ) - f.*uⁿ + Fᵛ
  Gʷⁿ = -u_dot_w(uⁿ, vⁿ, wⁿ) - g.* (δρ ./ ρ₀) + Fʷ

  # Calculate midpoint source terms using the Adams-Bashforth (AB2) method.
  @. begin
    Gᵘⁿ⁺ʰ = (3/2 + χ)*Gᵘⁿ - (1/2 + χ)*Gᵘⁿ⁻¹
    Gᵛⁿ⁺ʰ = (3/2 + χ)*Gᵛⁿ - (1/2 + χ)*Gᵛⁿ⁻¹
    Gʷⁿ⁺ʰ = (3/2 + χ)*Gʷⁿ - (1/2 + χ)*Gʷⁿ⁻¹
    Gᵀⁿ⁺ʰ = (3/2 + χ)*Gᵀⁿ - (1/2 + χ)*Gᵀⁿ⁻¹
    Gˢⁿ⁺ʰ = (3/2 + χ)*Gˢⁿ - (1/2 + χ)*Gˢⁿ⁻¹
  end

  pⁿ = solve_for_pressure(Gᵘⁿ⁺ʰ, Gᵛⁿ⁺ʰ, Gʷⁿ⁺ʰ)

  # Calculate non-hydrostatic component of pressure (as a residual from the
  # total pressure) for vertical velocity time-stepping.
  pⁿʰ = pⁿ .- pʰʸ

  uⁿ = uⁿ .+ (Gᵘⁿ⁺ʰ .- (Aˣ/V).*δˣ(pⁿ)) ./ Δt
  vⁿ = vⁿ .+ (Gᵛⁿ⁺ʰ .- (Aʸ/V).*δʸ(pⁿ)) ./ Δt
  wⁿ = wⁿ .+ (Gʷⁿ⁺ʰ .- (Aᶻ/V).*δᶻ(pⁿʰ)) ./ Δt
  Sⁿ = Sⁿ .+ Gˢⁿ⁺ʰ./Δt
  Tⁿ = Tⁿ .+ Gᵀⁿ⁺ʰ./Δt
end

# TODO: We can use Makie to create 3D OpenGL visualizations to debug the runs.
