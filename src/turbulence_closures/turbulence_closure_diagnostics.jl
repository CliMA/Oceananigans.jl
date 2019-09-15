# Timescale for diffusion across one cell
min_Δxyz(grid) = min(grid.Δx, grid.Δy, grid.Δz)
min_Δxy(grid) = min(grid.Δx, grid.Δy)
min_Δz(grid) = grid.Δz

"Returns the time-scale for diffusion on a regular grid across a single grid cell."
function cell_diffusion_timescale(model::AbstractModel{TS, <:IsotropicDiffusivity}) where TS
    Δ = min_Δxyz(model.grid)
    return min(Δ^2 / model.closure.ν, Δ^2 / model.closure.κ)
end

function cell_diffusion_timescale(model::AbstractModel{TS, <:TensorDiffusivity}) where TS
    Δh = min_Δxy(model.grid)
    Δz = min_Δz(model.grid)
    return min(Δz^2 / model.closure.νv, Δh^2 / model.closure.νh,
               Δz^2 / model.closure.κv, Δh^2 / model.closure.κh)
end

function cell_diffusion_timescale(model::AbstractModel{TS, <:AbstractSmagorinsky}) where TS
    Δ = min_Δxyz(model.grid)
    max_νκ = maximum(model.diffusivities.νₑ.data.parent) * max(1, 1/model.closure.Pr)
    return min(Δ^2 / max_νκ, Δ^2 / model.closure.κ)
end

function cell_diffusion_timescale(model::AbstractModel{TS, <:AbstractAnisotropicMinimumDissipation}) where TS
    Δ = min_Δxyz(model.grid)
    max_ν = maximum(model.diffusivities.νₑ.data.parent)
    max_κ = max(Tuple(maximum(κₑ.data.parent) for κₑ in model.diffusivities.κₑ)...)
    return min(Δ^2 / max_ν, Δ^2 / max_κ)
end

