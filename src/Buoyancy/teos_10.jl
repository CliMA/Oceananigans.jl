#####
##### The TEOS-10 polynomial approximation implemented in this file has been translated
##### into Julia from https://github.com/fabien-roquet/polyTEOS/blob/master/polyTEOS10.py
#####

struct TEOS10{FT} <: AbstractNonlinearEquationOfState end

#####
##### Reference values chosen using TEOS-10 recommendation
#####

const Sₐᵤ = 40 * 35.16504 / 35
const Θᵤ  = 40.0
const Zᵤ  = 1e4
const ΔS  = 32.0

#####
##### Coordinate transformations from (Θ, S, p) to (τ, s, ζ)
#####

@inline τ(Θ) = Θ / Θᵤ
@inline s(S) = √((S + ΔS) / Sₐᵤ)
@inline ζ(p) = p / Zᵤ

#####
##### Vertical reference profile of density
#####

const R₀₀ =  4.6494977072e+01
const R₀₁ = -5.2099962525e+00
const R₀₂ =  2.2601900708e-01
const R₀₃ =  6.4326772569e-02
const R₀₄ =  1.5616995503e-02
const R₀₅ = -1.7243708991e-03

@inline r₀(ζ) = (((((R₀₅ * ζ + R₀₄) * ζ + R₀₃) * ζ + R₀₂) * ζ + R₀₁) * ζ + R₀₀) * ζ

#####
##### Density anomaly fit
#####

const R₀₀₀ =  8.0189615746e+02
const R₁₀₀ =  8.6672408165e+02
const R₂₀₀ = -1.7864682637e+03
const R₃₀₀ =  2.0375295546e+03
const R₄₀₀ = -1.2849161071e+03
const R₅₀₀ =  4.3227585684e+02
const R₆₀₀ = -6.0579916612e+01
const R₀₁₀ =  2.6010145068e+01
const R₁₁₀ = -6.5281885265e+01
const R₂₁₀ =  8.1770425108e+01
const R₃₁₀ = -5.6888046321e+01
const R₄₁₀ =  1.7681814114e+01
const R₅₁₀ = -1.9193502195e+00
const R₀₂₀ = -3.7074170417e+01
const R₁₂₀ =  6.1548258127e+01
const R₂₂₀ = -6.0362551501e+01
const R₃₂₀ =  2.9130021253e+01
const R₄₂₀ = -5.4723692739e+00
const R₀₃₀ =  2.1661789529e+01
const R₁₃₀ = -3.3449108469e+01
const R₂₃₀ =  1.9717078466e+01
const R₃₃₀ = -3.1742946532e+00
const R₀₄₀ = -8.3627885467e+00
const R₁₄₀ =  1.1311538584e+01
const R₂₄₀ = -5.3563304045e+00
const R₀₅₀ =  5.4048723791e-01
const R₁₅₀ =  4.8169980163e-01
const R₀₆₀ = -1.9083568888e-01
const R₀₀₁ =  1.9681925209e+01
const R₁₀₁ = -4.2549998214e+01
const R₂₀₁ =  5.0774768218e+01
const R₃₀₁ = -3.0938076334e+01
const R₄₀₁ =  6.6051753097e+00
const R₀₁₁ = -1.3336301113e+01
const R₁₁₁ = -4.4870114575e+00
const R₂₁₁ =  5.0042598061e+00
const R₃₁₁ = -6.5399043664e-01
const R₀₂₁ =  6.7080479603e+00
const R₁₂₁ =  3.5063081279e+00
const R₂₂₁ = -1.8795372996e+00
const R₀₃₁ = -2.4649669534e+00
const R₁₃₁ = -5.5077101279e-01
const R₀₄₁ =  5.5927935970e-01
const R₀₀₂ =  2.0660924175e+00
const R₁₀₂ = -4.9527603989e+00
const R₂₀₂ =  2.5019633244e+00
const R₀₁₂ =  2.0564311499e+00
const R₁₁₂ = -2.1311365518e-01
const R₀₂₂ = -1.2419983026e+00
const R₀₀₃ = -2.3342758797e-02
const R₁₀₃ = -1.8507636718e-02
const R₀₁₃ =  3.7969820455e-01

@inline r′₃(τ, s) = R₀₁₃ * τ + R₁₀₃ * s + R₀₀₃

@inline r′₂(τ, s) = (R₀₂₂ * τ + R₁₁₂ * s + R₀₁₂) * τ + (R₂₀₂ * s + R₁₀₂) * s + R₀₀₂

@inline r′₁(τ, s) =
    (((R₀₄₁ * τ + R₁₃₁ * s + R₀₃₁) * τ +
      (R₂₂₁ * s + R₁₂₁) * s + R₀₂₁) * τ +
     ((R₃₁₁ * s + R₂₁₁) * s + R₁₁₁) * s + R₀₁₁) * τ +
    (((R₄₀₁ * s + R₃₀₁) * s + R₂₀₁) * s + R₁₀₁) * s + R₀₀₁

@inline r′₀(τ, s) =
    (((((R₀₆₀ * τ + R₁₅₀ * s + R₀₅₀) * τ +
        (R₂₄₀ * s + R₁₄₀) * s + R₀₄₀) * τ +
       ((R₃₃₀ * s + R₂₃₀) * s + R₁₃₀) * s + R₀₃₀) * τ +
      (((R₄₂₀ * s + R₃₂₀) * s + R₂₂₀) * s + R₁₂₀) * s + R₀₂₀) * τ +
     ((((R₅₁₀ * s + R₄₁₀) * s + R₃₁₀) * s + R₂₁₀) * s + R₁₁₀) * s + R₀₁₀) * τ +
    (((((R₆₀₀ * s + R₅₀₀) * s + R₄₀₀) * s + R₃₀₀) * s + R₂₀₀) * s + R₁₀₀) * s + R₀₀₀

@inline r′(τ, s, ζ) = ((r′₃(τ, s) * ζ + r′₂(τ, s)) * ζ + r′₁(τ, s)) * ζ + r′₀(τ, s)

#####
##### Full density
#####

"""
    ρ(Θ, S, p)

Returns the in-situ density of seawater with state (Θ, S, p) using the 55-term
polynomial approximation TEOS-10 described in Roquet et al. (§3.1, 2014).

# Inputs
    `Θ`: conservative temperature (ITS-90) [°C]
    `S`: absolute salinity [g/kg]
    `p`: sea pressure [dbar] (i.e. absolute pressure - 10.1325 dbar)

# Output
    `ρ`: in-situ density [kg/m³]

# References
Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate polynomial
    expressions for the density and specific volume of seawater using the TEOS-10
    standard. Ocean Modelling.
"""
@inline ρ(Θ, S, p) = _ρ(τ(Θ), s(S), ζ(p))

@inline _ρ(τ, s, ζ) = r₀(ζ) + r′(τ, s, ζ)

#####
##### Thermal expansion fit
#####

const α₀₀₀ = -6.5025362670e-01
const α₁₀₀ =  1.6320471316e+00
const α₂₀₀ = -2.0442606277e+00
const α₃₀₀ =  1.4222011580e+00
const α₄₀₀ = -4.4204535284e-01
const α₅₀₀ =  4.7983755487e-02
const α₀₁₀ =  1.8537085209e+00
const α₁₁₀ = -3.0774129064e+00
const α₂₁₀ =  3.0181275751e+00
const α₃₁₀ = -1.4565010626e+00
const α₄₁₀ =  2.7361846370e-01
const α₀₂₀ = -1.6246342147e+00
const α₁₂₀ =  2.5086831352e+00
const α₂₂₀ = -1.4787808849e+00
const α₃₂₀ =  2.3807209899e-01
const α₀₃₀ =  8.3627885467e-01
const α₁₃₀ = -1.1311538584e+00
const α₂₃₀ =  5.3563304045e-01
const α₀₄₀ = -6.7560904739e-02
const α₁₄₀ = -6.0212475204e-02
const α₀₅₀ =  2.8625353333e-02
const α₀₀₁ =  3.3340752782e-01
const α₁₀₁ =  1.1217528644e-01
const α₂₀₁ = -1.2510649515e-01
const α₃₀₁ =  1.6349760916e-02
const α₀₁₁ = -3.3540239802e-01
const α₁₁₁ = -1.7531540640e-01
const α₂₁₁ =  9.3976864981e-02
const α₀₂₁ =  1.8487252150e-01
const α₁₂₁ =  4.1307825959e-02
const α₀₃₁ = -5.5927935970e-02
const α₀₀₂ = -5.1410778748e-02
const α₁₀₂ =  5.3278413794e-03
const α₀₁₂ =  6.2099915132e-02
const α₀₀₃ = -9.4924551138e-03

"""
    α(Θ, S, p)

Returns the Boussinesq thermal expansion coefficient -∂ρ/∂Θ [kg/m³/K] computed using
the 55-term polynomial approximation to TEOS-10 described in Roquet et al. (§3.1, 2014).

# Inputs
    `Θ`: conservative temperature (ITS-90) [°C]
    `S`: absolute salinity [g/kg]
    `p`: sea pressure [dbar] (i.e. absolute pressure - 10.1325 dbar)

# Output
    `α`: Boussinesq thermal expansion coefficient -∂ρ/∂Θ [kg/m³/K]

# References
Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate polynomial
    expressions for the density and specific volume of seawater using the TEOS-10
    standard. Ocean Modelling.
"""
@inline α(Θ, S, p) = _α(τ(Θ), s(S), ζ(p))

@inline _α(τ, s, ζ) =
    ((α₀₀₃ * ζ + α₀₁₂ * τ + α₁₀₂ * s + α₀₀₂) * ζ +
     ((α₀₃₁ * τ + α₁₂₁ * s + α₀₂₁) * τ +
      (α₂₁₁ * s + α₁₁₁) * s + α₀₁₁) * τ +
     ((α₃₀₁ * s + α₂₀₁) * s + α₁₀₁) * s + α₀₀₁) * ζ +
    ((((α₀₅₀ * τ + α₁₄₀ * s + α₀₄₀) * τ +
       (α₂₃₀ * s + α₁₃₀) * s + α₀₃₀) * τ +
      ((α₃₂₀ * s + α₂₂₀) * s + α₁₂₀) * s + α₀₂₀) * τ +
     (((α₄₁₀ * s + α₃₁₀) * s + α₂₁₀) * s + α₁₁₀) * s + α₀₁₀) * τ +
    ((((α₅₀₀ * s + α₄₀₀) * s + α₃₀₀) * s + α₂₀₀) * s + α₁₀₀) * s + α₀₀₀

#####
##### Saline contraction
#####

const β₀₀₀ =  1.0783203594e+01
const β₁₀₀ = -4.4452095908e+01
const β₂₀₀ =  7.6048755820e+01
const β₃₀₀ = -6.3944280668e+01
const β₄₀₀ =  2.6890441098e+01
const β₅₀₀ = -4.5221697773e+00
const β₀₁₀ = -8.1219372432e-01
const β₁₁₀ =  2.0346663041e+00
const β₂₁₀ = -2.1232895170e+00
const β₃₁₀ =  8.7994140485e-01
const β₄₁₀ = -1.1939638360e-01
const β₀₂₀ =  7.6574242289e-01
const β₁₂₀ = -1.5019813020e+00
const β₂₂₀ =  1.0872489522e+00
const β₃₂₀ = -2.7233429080e-01
const β₀₃₀ = -4.1615152308e-01
const β₁₃₀ =  4.9061350869e-01
const β₂₃₀ = -1.1847737788e-01
const β₀₄₀ =  1.4073062708e-01
const β₁₄₀ = -1.3327978879e-01
const β₀₅₀ =  5.9929880134e-03
const β₀₀₁ = -5.2937873009e-01
const β₁₀₁ =  1.2634116779e+00
const β₂₀₁ = -1.1547328025e+00
const β₃₀₁ =  3.2870876279e-01
const β₀₁₁ = -5.5824407214e-02
const β₁₁₁ =  1.2451933313e-01
const β₂₁₁ = -2.4409539932e-02
const β₀₂₁ =  4.3623149752e-02
const β₁₂₁ = -4.6767901790e-02
const β₀₃₁ = -6.8523260060e-03
const β₀₀₂ = -6.1618945251e-02
const β₁₀₂ =  6.2255521644e-02
const β₀₁₂ = -2.6514181169e-03
const β₀₀₃ = -2.3025968587e-04

"""
    β(Θ, S, p)

Returns the Boussinesq haline contraction coefficient ∂ρ/∂S [kg/m³/(g/kg)] computed using
the 55-term polynomial approximation to TEOS-10 described in Roquet et al. (§3.1, 2014).

# Inputs
    `Θ`: conservative temperature (ITS-90) [°C]
    `S`: absolute salinity [g/kg]
    `p`: sea pressure [dbar] (i.e. absolute pressure - 10.1325 dbar)

# Output
    `β`: Boussinesq haline contraction coefficient ∂ρ/∂S [kg/m³/(g/kg)]

# References
Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate polynomial
    expressions for the density and specific volume of seawater using the TEOS-10
    standard. Ocean Modelling.
"""
@inline β(Θ, S, p) = _β(τ(Θ), s(S), ζ(p)) / s(S)

@inline _β(τ, s, ζ) =
    ((β₀₀₃ * ζ + β₀₁₂ * τ + β₁₀₂ * s + β₀₀₂) * ζ +
     ((β₀₃₁ * τ + β₁₂₁ * s + β₀₂₁) * τ +
      (β₂₁₁ * s + β₁₁₁) * s + β₀₁₁) * τ +
     ((β₃₀₁ * s + β₂₀₁) * s + β₁₀₁) * s + β₀₀₁) * ζ +
    ((((β₀₅₀ * τ + β₁₄₀ * s + β₀₄₀) * τ +
       (β₂₃₀ * s + β₁₃₀) * s + β₀₃₀) * τ +
      ((β₃₂₀ * s + β₂₂₀) * s + β₁₂₀) * s + β₀₂₀) * τ +
     (((β₄₁₀ * s + β₃₁₀) * s + β₂₁₀) * s + β₁₁₀) * s + β₀₁₀) * τ +
    ((((β₅₀₀ * s + β₄₀₀) * s + β₃₀₀) * s + β₂₀₀) * s + β₁₀₀) * s + β₀₀₀
