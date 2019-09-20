# Timescale for diffusion across one cell
min_Δxyz(grid) = min(grid.Δx, grid.Δy, grid.Δz)
min_Δxy(grid) = min(grid.Δx, grid.Δy)
min_Δz(grid) = grid.Δz

cell_diffusion_timescale(model) = cell_diffusion_timescale(model.closure, model.diffusivities, model.grid)

"Returns the time-scale for diffusion on a regular grid across a single grid cell."
function cell_diffusion_timescale(closure::IsotropicDiffusivity, diffusivities, grid)
    Δ = min_Δxyz(grid)
    return min(Δ^2 / closure.ν, Δ^2 / closure.κ)
end

function cell_diffusion_timescale(closure::TensorDiffusivity, diffusivies, grid)
    Δh = min_Δxy(grid)
    Δz = min_Δz(grid)
    return min(Δz^2 / closure.νv, Δh^2 / closure.νh,
               Δz^2 / closure.κv, Δh^2 / closure.κh)
end

function cell_diffusion_timescale(closure::AbstractSmagorinsky, diffusivies, grid)
    Δ = min_Δxyz(grid)
    max_νκ = maximum(diffusivities.νₑ.data.parent) * max(1, 1/closure.Pr)
    return min(Δ^2 / max_νκ, Δ^2 / closure.κ)
end

function cell_diffusion_timescale(closure::AbstractAnisotropicMinimumDissipation, diffusivies, grid)
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in diffusivities.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

