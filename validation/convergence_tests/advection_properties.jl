#using Oceananigans.Advection

rate_of_convergence(::CenteredSecondOrder)    = 2
rate_of_convergence(::UpwindBiasedThirdOrder) = 3
rate_of_convergence(::CenteredFourthOrder)    = 4
rate_of_convergence(::UpwindBiasedFifthOrder) = 5
rate_of_convergence(::WENO5)                  = 5

labels(::CenteredSecondOrder)    = "Center2ⁿᵈ"
labels(::UpwindBiasedThirdOrder) = "Upwind3ʳᵈ"
labels(::CenteredFourthOrder)    = "Center4ᵗʰ"
labels(::UpwindBiasedFifthOrder) = "Upwind5ᵗʰ"
labels(::WENO5)                  = "WENO5ᵗʰ "

shapes(::CenteredSecondOrder)    = :diamond
shapes(::UpwindBiasedThirdOrder) = :dtriangle
shapes(::CenteredFourthOrder)    = :rect
shapes(::UpwindBiasedFifthOrder) = :star5
shapes(::WENO5)                  = :star6

colors(::CenteredSecondOrder)    = :green
colors(::UpwindBiasedThirdOrder) = :red
colors(::CenteredFourthOrder)    = :cyan
colors(::UpwindBiasedFifthOrder) = :magenta
colors(::WENO5)                  = :purple

halos(::CenteredSecondOrder)    = 1
halos(::UpwindBiasedThirdOrder) = 2
halos(::CenteredFourthOrder)    = 2
halos(::UpwindBiasedFifthOrder) = 3
halos(::WENO5)                  = 3