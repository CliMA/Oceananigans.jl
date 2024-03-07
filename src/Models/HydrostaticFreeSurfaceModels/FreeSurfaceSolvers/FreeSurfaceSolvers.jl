module FreeSurfaceSolvers

export ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface

include("explicit_free_surface.jl")
include("implicit_free_surface.jl")
include("implicit_free_surface_utils.jl")
include("pcg_implicit_free_surface_solver.jl")
include("matrix_implicit_free_surface_solver.jl")
include("split_explicit_free_surface.jl")
include("split_explicit_free_surface_kernels.jl")

end