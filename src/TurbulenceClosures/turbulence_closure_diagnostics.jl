# Timescale for diffusion across one cell
min_Δxyz(grid) = min(grid.Δx, grid.Δy, grid.Δz)
min_Δxy(grid) = min(grid.Δx, grid.Δy)

min_Δx(grid) = grid.Δx
min_Δy(grid) = grid.Δy
min_Δz(grid) = grid.Δz

cell_diffusion_timescale(model) = cell_diffusion_timescale(model.closure, model.diffusivities, model.grid)

function cell_diffusion_timescale(closure::IsotropicDiffusivity{N, <:NamedTuple{()}},
                                  diffusivities, grid) where N
    return min_Δxyz(grid)^2 / closure.ν
end

function cell_diffusion_timescale(closure::IsotropicDiffusivity, diffusivities, grid)
    Δ = min_Δxyz(grid)
    max_κ = maximum(closure.κ)
    return min(Δ^2 / closure.ν, Δ^2 / max_κ)
end

function cell_diffusion_timescale(closure::AnisotropicDiffusivity{NX, NY, NZ, <:NamedTuple{()}, <:NamedTuple{()}, <:NamedTuple{()}},
                                  diffusivities, grid) where {NX, NY, NZ}
    Δx = min_Δx(grid)
    Δy = min_Δy(grid)
    Δz = min_Δz(grid)

    return min(Δx^2 / closure.νx,
               Δy^2 / closure.νy,
               Δz^2 / closure.νz)
end

function cell_diffusion_timescale(closure::AnisotropicDiffusivity, diffusivities, grid)

    Δx = min_Δx(grid)
    Δy = min_Δy(grid)
    Δz = min_Δz(grid)

    max_κx = maximum(closure.κx)
    max_κy = maximum(closure.κy)
    max_κz = maximum(closure.κz)

    return min(Δx^2 / closure.νx,
               Δy^2 / closure.νy,
               Δz^2 / closure.νz,
               Δx^2 / max_κx,
               Δy^2 / max_κy,
               Δz^2 / max_κz)
end

function cell_diffusion_timescale(closure::AnisotropicBiharmonicDiffusivity{V, <:NamedTuple{()}, <:NamedTuple{()}},
                                  diffusivities, grid) where V
    Δx = min_Δx(grid)
    Δy = min_Δy(grid)
    Δz = min_Δz(grid)

    return min(Δx^4 / closure.νx,
               Δy^4 / closure.νy,
               Δz^4 / closure.νz)
end

function cell_diffusion_timescale(closure::AnisotropicBiharmonicDiffusivity, diffusivities, grid)
    Δx = min_Δx(grid)
    Δy = min_Δy(grid)
    Δz = min_Δz(grid)

    max_κx = maximum(closure.κx)
    max_κy = maximum(closure.κy)
    max_κz = maximum(closure.κz)

    return min(Δx^4 / closure.νx, 
               Δy^4 / closure.νy, 
               Δz^4 / closure.νz,
               Δx^4 / max_κx,
               Δy^4 / max_κy, 
               Δz^4 / max_κz)
end

function cell_diffusion_timescale(closure::SmagorinskyLilly{FT, P, <:NamedTuple{()}},
                                  diffusivities, grid) where {FT, P}
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    return Δ^2 / max_ν
end

function cell_diffusion_timescale(closure::BlasiusSmagorinsky{ML, FT, P, <:NamedTuple{()}},
                                  diffusivities, grid) where {ML, FT, P}
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    return Δ^2 / max_ν
end

function cell_diffusion_timescale(closure::AbstractSmagorinsky, diffusivities, grid)
    Δ = min_Δxyz(grid)
    min_Pr = minimum(closure.Pr)
    max_κ = maximum(closure.κ)
    max_νκ = maximum(diffusivities.νₑ.data.parent) * max(1, 1/min_Pr)
    return min(Δ^2 / max_νκ, Δ^2 / max_κ)
end

function cell_diffusion_timescale(closure::RozemaAnisotropicMinimumDissipation{FT, <:NamedTuple{()}},
                                  diffusivities, grid) where FT
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    return Δ^2 / max_ν
end

function cell_diffusion_timescale(closure::VerstappenAnisotropicMinimumDissipation{FT, PK, PN, <:NamedTuple{()}},
                                  diffusivities, grid) where {FT, PK, PN}
    Δ = min_Δxyz(grid)
    max_ν = maximum(diffusivities.νₑ.data.parent)
    return Δ^2 / max_ν
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

cell_diffusion_timescale(closure::Tuple, diffusivities, grid) =
    min(Tuple(cell_diffusion_timescale(c, diffusivities, grid) for c in closure)...)
