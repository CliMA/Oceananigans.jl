# Timescale for diffusion across one cell
min_Δxyz(grid) = min(grid.Δx, grid.Δy, grid.Δz)
min_Δxy(grid) = min(grid.Δx, grid.Δy)
min_Δz(grid) = grid.Δz

cell_diffusion_timescale(model) = cell_diffusion_timescale(model.closure, model.diffusivities, model.grid)

"Returns the time-scale for diffusion on a regular grid across a single grid cell."
function cell_diffusion_timescale(closure::ConstantIsotropicDiffusivity, diffusivities, grid)
    Δ = min_Δxyz(grid)
    max_κ = maximum(closure.κ)
    return min(Δ^2 / closure.ν, Δ^2 / max_κ)
end

function cell_diffusion_timescale(closure::ConstantAnisotropicDiffusivity, diffusivies, grid)
    Δh = min_Δxy(grid)
    Δz = min_Δz(grid)
    max_κh = maximum(closure.κh)
    max_κv = maximum(closure.κv)
    return min(Δz^2 / closure.νv, Δh^2 / closure.νh,
               Δz^2 / max_κv, Δh^2 / max_κh)
end

function cell_diffusion_timescale(closure::AnisotropicBiharmonicDiffusivity, diffusivies, grid)
    Δh = min_Δxy(grid)
    Δz = min_Δz(grid)
    max_κh = maximum(closure.κh)
    max_κv = maximum(closure.κv)
    return min(Δz^4 / closure.νv, Δh^4 / closure.νh,
               Δz^4 / max_κv, Δh^4 / max_κh)
end

function cell_diffusion_timescale(closure::Union{AbstractSmagorinsky, AbstractLeith}, diffusivies, grid)
    Δ = min_Δxyz(grid)
    min_Pr = minimum(closure.Pr)
    max_κ = maximum(closure.κ)
    max_νκ = maximum(diffusivities.νₑ.data.parent) * max(1, 1/min_Pr)
    return min(Δ^2 / max_νκ, Δ^2 / max_κ)
end

function cell_diffusion_timescale(closure::AbstractAnisotropicMinimumDissipation, diffusivies, grid)
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in diffusivities.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

