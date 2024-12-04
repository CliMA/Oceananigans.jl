module Operators

# Spacings
export Δxᶠᶠᶠ, Δxᶠᶠᶜ, Δxᶠᶜᶠ, Δxᶠᶜᶜ, Δxᶜᶠᶠ, Δxᶜᶠᶜ, Δxᶜᶜᶠ, Δxᶜᶜᶜ
export Δyᶠᶠᶠ, Δyᶠᶠᶜ, Δyᶠᶜᶠ, Δyᶠᶜᶜ, Δyᶜᶠᶠ, Δyᶜᶠᶜ, Δyᶜᶜᶠ, Δyᶜᶜᶜ
export Δzᶠᶠᶠ, Δzᶠᶠᶜ, Δzᶠᶜᶠ, Δzᶠᶜᶜ, Δzᶜᶠᶠ, Δzᶜᶠᶜ, Δzᶜᶜᶠ, Δzᶜᶜᶜ
export Δrᶠᶠᶠ, Δrᶠᶠᶜ, Δrᶠᶜᶠ, Δrᶠᶜᶜ, Δrᶜᶠᶠ, Δrᶜᶠᶜ, Δrᶜᶜᶠ, Δrᶜᶜᶜ

# Areas
export Axᶠᶠᶠ, Axᶠᶠᶜ, Axᶠᶜᶠ, Axᶠᶜᶜ, Axᶜᶠᶠ, Axᶜᶠᶜ, Axᶜᶜᶠ, Axᶜᶜᶜ
export Ayᶠᶠᶠ, Ayᶠᶠᶜ, Ayᶠᶜᶠ, Ayᶠᶜᶜ, Ayᶜᶠᶠ, Ayᶜᶠᶜ, Ayᶜᶜᶠ, Ayᶜᶜᶜ
export Azᶠᶠᶠ, Azᶠᶠᶜ, Azᶠᶜᶠ, Azᶠᶜᶜ, Azᶜᶠᶠ, Azᶜᶠᶜ, Azᶜᶜᶠ, Azᶜᶜᶜ

export Axᵃᶜᶜ, Axᵃᶠᶠ, Axᶜᵃᶜ, Axᶠᵃᶠ, Axᶜᶜᵃ, Axᶠᶠᵃ
export Ayᵃᶜᶜ, Ayᵃᶠᶠ, Ayᶜᵃᶜ, Ayᶠᵃᶠ, Ayᶜᶜᵃ, Ayᶠᶠᵃ
export Azᵃᶜᶜ, Azᵃᶠᶠ, Azᶜᵃᶜ, Azᶠᵃᶠ, Azᶜᶜᵃ, Azᶠᶠᵃ

# Volumes
export Vᶠᶠᶠ, Vᶠᶠᶜ, Vᶠᶜᶠ, Vᶠᶜᶜ, Vᶜᶠᶠ, Vᶜᶠᶜ, Vᶜᶜᶠ, Vᶜᶜᶜ

# Product between spacings and fields
export Δx_qᶠᶠᶠ, Δx_qᶠᶠᶜ, Δx_qᶠᶜᶠ, Δx_qᶠᶜᶜ, Δx_qᶜᶠᶠ, Δx_qᶜᶠᶜ, Δx_qᶜᶜᶠ, Δx_qᶜᶜᶜ
export Δy_qᶠᶠᶠ, Δy_qᶠᶠᶜ, Δy_qᶠᶜᶠ, Δy_qᶠᶜᶜ, Δy_qᶜᶠᶠ, Δy_qᶜᶠᶜ, Δy_qᶜᶜᶠ, Δy_qᶜᶜᶜ
export Δz_qᶠᶠᶠ, Δz_qᶠᶠᶜ, Δz_qᶠᶜᶠ, Δz_qᶠᶜᶜ, Δz_qᶜᶠᶠ, Δz_qᶜᶠᶜ, Δz_qᶜᶜᶠ, Δz_qᶜᶜᶜ

# Product between areas and fields
export Ax_qᶠᶠᶠ, Ax_qᶠᶠᶜ, Ax_qᶠᶜᶠ, Ax_qᶠᶜᶜ, Ax_qᶜᶠᶠ, Ax_qᶜᶠᶜ, Ax_qᶜᶜᶠ, Ax_qᶜᶜᶜ
export Ay_qᶠᶠᶠ, Ay_qᶠᶠᶜ, Ay_qᶠᶜᶠ, Ay_qᶠᶜᶜ, Ay_qᶜᶠᶠ, Ay_qᶜᶠᶜ, Ay_qᶜᶜᶠ, Ay_qᶜᶜᶜ
export Az_qᶠᶠᶠ, Az_qᶠᶠᶜ, Az_qᶠᶜᶠ, Az_qᶠᶜᶜ, Az_qᶜᶠᶠ, Az_qᶜᶠᶜ, Az_qᶜᶜᶠ, Az_qᶜᶜᶜ

# Differences
export δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ
export δxᶠᶠᶠ, δxᶠᶠᶜ, δxᶠᶜᶠ, δxᶠᶜᶜ, δxᶜᶠᶠ, δxᶜᶠᶜ, δxᶜᶜᶠ, δxᶜᶜᶜ
export δyᶠᶠᶠ, δyᶠᶠᶜ, δyᶠᶜᶠ, δyᶠᶜᶜ, δyᶜᶠᶠ, δyᶜᶠᶜ, δyᶜᶜᶠ, δyᶜᶜᶜ
export δzᶠᶠᶠ, δzᶠᶠᶜ, δzᶠᶜᶠ, δzᶠᶜᶜ, δzᶜᶠᶠ, δzᶜᶠᶜ, δzᶜᶜᶠ, δzᶜᶜᶜ

# Derivatives
export ∂xᶜᵃᵃ, ∂xᶠᵃᵃ, ∂yᵃᶜᵃ, ∂yᵃᶠᵃ, ∂zᵃᵃᶜ, ∂zᵃᵃᶠ

export ∂xᶠᶠᶠ, ∂xᶠᶠᶜ, ∂xᶠᶜᶠ, ∂xᶠᶜᶜ, ∂xᶜᶠᶠ, ∂xᶜᶠᶜ, ∂xᶜᶜᶠ, ∂xᶜᶜᶜ
export ∂yᶠᶠᶠ, ∂yᶠᶠᶜ, ∂yᶠᶜᶠ, ∂yᶠᶜᶜ, ∂yᶜᶠᶠ, ∂yᶜᶠᶜ, ∂yᶜᶜᶠ, ∂yᶜᶜᶜ
export ∂zᶠᶠᶠ, ∂zᶠᶠᶜ, ∂zᶠᶜᶠ, ∂zᶠᶜᶜ, ∂zᶜᶠᶠ, ∂zᶜᶠᶜ, ∂zᶜᶜᶠ, ∂zᶜᶜᶜ

export ∂²xᶠᶠᶠ, ∂²xᶠᶠᶜ, ∂²xᶠᶜᶠ, ∂²xᶠᶜᶜ, ∂²xᶜᶠᶠ, ∂²xᶜᶠᶜ, ∂²xᶜᶜᶠ, ∂²xᶜᶜᶜ
export ∂²yᶠᶠᶠ, ∂²yᶠᶠᶜ, ∂²yᶠᶜᶠ, ∂²yᶠᶜᶜ, ∂²yᶜᶠᶠ, ∂²yᶜᶠᶜ, ∂²yᶜᶜᶠ, ∂²yᶜᶜᶜ
export ∂²zᶠᶠᶠ, ∂²zᶠᶠᶜ, ∂²zᶠᶜᶠ, ∂²zᶠᶜᶜ, ∂²zᶜᶠᶠ, ∂²zᶜᶠᶜ, ∂²zᶜᶜᶠ, ∂²zᶜᶜᶜ

export ∂³xᶠᶠᶠ, ∂³xᶠᶠᶜ, ∂³xᶠᶜᶠ, ∂³xᶠᶜᶜ, ∂³xᶜᶠᶠ, ∂³xᶜᶠᶜ, ∂³xᶜᶜᶠ, ∂³xᶜᶜᶜ
export ∂³yᶠᶠᶠ, ∂³yᶠᶠᶜ, ∂³yᶠᶜᶠ, ∂³yᶠᶜᶜ, ∂³yᶜᶠᶠ, ∂³yᶜᶠᶜ, ∂³yᶜᶜᶠ, ∂³yᶜᶜᶜ
export ∂³zᶠᶠᶠ, ∂³zᶠᶠᶜ, ∂³zᶠᶜᶠ, ∂³zᶠᶜᶜ, ∂³zᶜᶠᶠ, ∂³zᶜᶠᶜ, ∂³zᶜᶜᶠ, ∂³zᶜᶜᶜ

export ∂⁴xᶠᶠᶠ, ∂⁴xᶠᶠᶜ, ∂⁴xᶠᶜᶠ, ∂⁴xᶠᶜᶜ, ∂⁴xᶜᶠᶠ, ∂⁴xᶜᶠᶜ, ∂⁴xᶜᶜᶠ, ∂⁴xᶜᶜᶜ
export ∂⁴yᶠᶠᶠ, ∂⁴yᶠᶠᶜ, ∂⁴yᶠᶜᶠ, ∂⁴yᶠᶜᶜ, ∂⁴yᶜᶠᶠ, ∂⁴yᶜᶠᶜ, ∂⁴yᶜᶜᶠ, ∂⁴yᶜᶜᶜ
export ∂⁴zᶠᶠᶠ, ∂⁴zᶠᶠᶜ, ∂⁴zᶠᶜᶠ, ∂⁴zᶠᶜᶜ, ∂⁴zᶜᶠᶠ, ∂⁴zᶜᶠᶜ, ∂⁴zᶜᶜᶠ, ∂⁴zᶜᶜᶜ

# Product between areas and derivatives
export Ax_∂xᶠᶠᶠ, Ax_∂xᶠᶠᶜ, Ax_∂xᶠᶜᶠ, Ax_∂xᶠᶜᶜ, Ax_∂xᶜᶠᶠ, Ax_∂xᶜᶠᶜ, Ax_∂xᶜᶜᶠ, Ax_∂xᶜᶜᶜ
export Ay_∂yᶠᶠᶠ, Ay_∂yᶠᶠᶜ, Ay_∂yᶠᶜᶠ, Ay_∂yᶠᶜᶜ, Ay_∂yᶜᶠᶠ, Ay_∂yᶜᶠᶜ, Ay_∂yᶜᶜᶠ, Ay_∂yᶜᶜᶜ
export Az_∂zᶠᶠᶠ, Az_∂zᶠᶠᶜ, Az_∂zᶠᶜᶠ, Az_∂zᶠᶜᶜ, Az_∂zᶜᶠᶠ, Az_∂zᶜᶠᶜ, Az_∂zᶜᶜᶠ, Az_∂zᶜᶜᶜ

# Divergences
export divᶜᶜᶜ, div_xyᶜᶜᶜ, div_xyᶜᶜᶠ, ζ₃ᶠᶠᶜ
export ∇²ᶜᶜᶜ, ∇²ᶠᶜᶜ, ∇²ᶜᶠᶜ, ∇²ᶜᶜᶠ, ∇²hᶜᶜᶜ, ∇²hᶠᶜᶜ, ∇²hᶜᶠᶜ

# Interpolations
export ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
export ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, ℑyzᵃᶜᶠ
export ℑxyzᶜᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶠ, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ

# Topology-aware operators
export δxTᶠᵃᵃ, δyTᵃᶠᵃ, δxTᶜᵃᵃ, δyTᵃᶜᵃ
export ∂xTᶠᶜᶠ, ∂yTᶜᶠᶠ

# Reference frame conversion
export intrinsic_vector, extrinsic_vector

using Oceananigans.Grids

import Oceananigans.Grids: xspacing, yspacing, zspacing

#####
##### Convenient aliases
#####

const AG = AbstractGrid

const Δx = xspacing
const Δy = yspacing
const Δz = zspacing

const RG  = RectilinearGrid
const RGX = XRegularRG
const RGY = YRegularRG
const RGZ = ZRegularRG

const OSSG  = OrthogonalSphericalShellGrid
const OSSGZ = ZRegOrthogonalSphericalShellGrid

const LLG  = LatitudeLongitudeGrid
const LLGX = XRegularLLG
const LLGY = YRegularLLG
const LLGZ = ZRegularLLG

# On the fly calculations of metrics
const LLGF  = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Nothing}
const LLGFX = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Number}
const LLGFY = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Nothing, <:Any, <:Any, <:Number}

include("difference_operators.jl")
include("interpolation_operators.jl")
include("interpolation_utils.jl")

include("spacings_and_areas_and_volumes.jl")
include("products_between_fields_and_grid_metrics.jl")

include("derivative_operators.jl")
include("divergence_operators.jl")
include("topology_aware_operators.jl")
include("vorticity_operators.jl")
include("laplacian_operators.jl")

include("variable_grid_operators.jl")
include("vector_rotation_operators.jl")

end # module
