# Timescale for diffusion across one cell
min_Δxyz(grid) = min(grid.Δx, grid.Δy, grid.Δz)
min_Δxy(grid) = min(grid.Δx, grid.Δy)
min_Δz(grid) = grid.Δz

cell_diffusion_timescale(model) = cell_diffusion_timescale(model.closure, model.diffusivities, model.grid)

function cell_diffusion_timescale(closure::ConstantIsotropicDiffusivity{V, <:NamedTuple{()}},
                                  diffusivities, grid) where V
    return min_Δxyz(grid)^2 / closure.ν
end

function cell_diffusion_timescale(closure::ConstantIsotropicDiffusivity, diffusivities, grid)
    Δ = min_Δxyz(grid)
    max_κ = maximum(closure.κ)
    return min(Δ^2 / closure.ν, Δ^2 / max_κ)
end

function cell_diffusion_timescale(closure::ConstantAnisotropicDiffusivity{V, <:NamedTuple{()}, <:NamedTuple{()}},
                                  diffusivities, grid) where V
    Δh = min_Δxy(grid)
    Δz = min_Δz(grid)
    return min(Δh^2 / closure.νh, Δz^2 / closure.νv)
end

function cell_diffusion_timescale(closure::ConstantAnisotropicDiffusivity, diffusivities, grid)
    Δh = min_Δxy(grid)
    Δz = min_Δz(grid)
    max_κh = maximum(closure.κh)
    max_κv = maximum(closure.κv)
    return min(Δh^2 / closure.νh, Δz^2 / closure.νv,
               Δh^2 / max_κh, Δz^2 / max_κv)
end

function cell_diffusion_timescale(closure::AnisotropicBiharmonicDiffusivity{V, <:NamedTuple{()}, <:NamedTuple{()}},
                                  diffusivities, grid) where V
    Δh = min_Δxy(grid)
    Δz = min_Δz(grid)
    return min(Δh^4 / closure.νh, Δz^4 / closure.νv)
end

function cell_diffusion_timescale(closure::AnisotropicBiharmonicDiffusivity, diffusivities, grid)
    Δh = min_Δxy(grid)
    Δz = min_Δz(grid)
<<<<<<< HEAD
    if length(closure.κh) == 0
        return min(Δz^4 / closure.νv, Δh^4 / closure.νh)
    else
        max_κh = maximum(closure.κh)
        max_κv = maximum(closure.κv)
        return min(Δz^4 / closure.νv, Δh^4 / closure.νh,
                   Δz^4 / max_κv, Δh^4 / max_κh)
    end
=======
    max_κh = maximum(closure.κh)
    max_κv = maximum(closure.κv)
    return min(Δh^4 / closure.νh, Δz^4 / closure.νv,
               Δh^4 / max_κh, Δz^4 / max_κv)
>>>>>>> dispatch on whether `closure.κ` is an empty named tuple or not
end

function cell_diffusion_timescale(closure::AbstractSmagorinsky, diffusivities, grid)
    Δ = min_Δxyz(grid)
    min_Pr = minimum(closure.Pr)
    max_κ = maximum(closure.κ)
    max_νκ = maximum(diffusivities.νₑ.data.parent) * max(1, 1/min_Pr)
    return min(Δ^2 / max_νκ, Δ^2 / max_κ)
end

function cell_diffusion_timescale(closure::AbstractAnisotropicMinimumDissipation, diffusivities, grid)
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in diffusivities.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

function cell_diffusion_timescale(closure::AbstractLeith, diffusivities, grid)
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    return Δ^2 / max_ν
end
