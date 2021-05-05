abstract type AbstractTimeDiscretization end

struct ExplicitDiscretization <: AbstractTimeDiscretization end

struct VerticallyImplicitDiscretization <: AbstractTimeDiscretization end

time_discretization(closure) = ExplicitDiscretization() # fallback

