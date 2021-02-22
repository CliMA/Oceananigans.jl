"""
    correct_immersed_tendencies!(model, Δt, γⁿ, ζⁿ)

Change the tendency fields to account for the presence of a boundary immersed
within the `model` grid. Does nothing by default.
"""
correct_immersed_tendencies!(model, Δt, γⁿ, ζⁿ) = nothing # fallback function 
