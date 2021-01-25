using Oceananigans.Advection

#rate_of_convergence(::UpwindBiasedFirstOrder) = 1
rate_of_convergence(::CenteredSecondOrder)    = 2
rate_of_convergence(::UpwindBiasedThirdOrder) = 3
rate_of_convergence(::CenteredFourthOrder)    = 4
rate_of_convergence(::UpwindBiasedFifthOrder) = 5
#rate_of_convergence(::CenteredSixthOrder)     = 6

#labels(::UpwindBiasedFirstOrder) = "Upwind1ˢᵗ"
labels(::CenteredSecondOrder)    = "Center2ⁿᵈ"
labels(::UpwindBiasedThirdOrder) = "Upwind3ʳᵈ"
labels(::CenteredFourthOrder)    = "Center4ᵗʰ"
labels(::UpwindBiasedFifthOrder) = "Upwind5ᵗʰ"
#labels(::CenteredSixthOrder)     = "Center6ᵗʰ"

#shapes(::UpwindBiasedFirstOrder) = :circle
shapes(::CenteredSecondOrder)    = :diamond
shapes(::UpwindBiasedThirdOrder) = :dtriangle
shapes(::CenteredFourthOrder)    = :rect
shapes(::UpwindBiasedFifthOrder) = :star5
#shapes(::CenteredSixthOrder)     = :star6

#colors(::UpwindBiasedFirstOrder) = :blue
colors(::CenteredSecondOrder)    = :green
colors(::UpwindBiasedThirdOrder) = :red
colors(::CenteredFourthOrder)    = :cyan
colors(::UpwindBiasedFifthOrder) = :magenta
#colors(::CenteredSixthOrder)     = :purple

#halos(::UpwindBiasedFirstOrder) = 1
halos(::CenteredSecondOrder)    = 1
halos(::UpwindBiasedThirdOrder) = 2
halos(::CenteredFourthOrder)    = 2
halos(::UpwindBiasedFifthOrder) = 3
#halos(::CenteredSixthOrder)     = 3





