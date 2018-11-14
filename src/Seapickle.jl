module Seapickle

export
  δˣ,
  δʸ,
  δᶻ,
  avgˣ,
  avgʸ,
  avgᶻ,
  div,
  div_flux,
  u_dot_u,
  u_dot_v,
  u_dot_w,
  laplacian_diffusion_zone,
  laplacian_diffusion_face_h,
  laplacian_diffusion_face_v,

  ρ₀,
  T₀,
  S₀,
  p₀,
  βᵖ,
  βˢ,
  βᵀ,
  αᵥ,
  ρ,

  solve_for_pressure,

using
  FFTW

include("operators.jl")
include("equation_of_state.jl")
include("pressure_solve.jl")

end
