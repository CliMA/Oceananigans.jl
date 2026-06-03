module Operators

# Some operators are exported next to where they are defined
# Spacings, Areas, and Volumes and their reciprocals: src/Operators/spacings_and_areas_and_volumes.jl
# Differences and Derivatives: src/Operators/derivative_operators.jl

# General metric operators
export xspacing, yspacing, zspacing, λspacing, φspacing, xarea, yarea, zarea, volume

# Product between spacings and fields
export Δx_qᶠᶠᶠ, Δx_qᶠᶠᶜ, Δx_qᶠᶜᶠ, Δx_qᶠᶜᶜ, Δx_qᶜᶠᶠ, Δx_qᶜᶠᶜ, Δx_qᶜᶜᶠ, Δx_qᶜᶜᶜ
export Δy_qᶠᶠᶠ, Δy_qᶠᶠᶜ, Δy_qᶠᶜᶠ, Δy_qᶠᶜᶜ, Δy_qᶜᶠᶠ, Δy_qᶜᶠᶜ, Δy_qᶜᶜᶠ, Δy_qᶜᶜᶜ
export Δz_qᶠᶠᶠ, Δz_qᶠᶠᶜ, Δz_qᶠᶜᶠ, Δz_qᶠᶜᶜ, Δz_qᶜᶠᶠ, Δz_qᶜᶠᶜ, Δz_qᶜᶜᶠ, Δz_qᶜᶜᶜ

# Product between areas and fields
export Ax_qᶠᶠᶠ, Ax_qᶠᶠᶜ, Ax_qᶠᶜᶠ, Ax_qᶠᶜᶜ, Ax_qᶜᶠᶠ, Ax_qᶜᶠᶜ, Ax_qᶜᶜᶠ, Ax_qᶜᶜᶜ
export Ay_qᶠᶠᶠ, Ay_qᶠᶠᶜ, Ay_qᶠᶜᶠ, Ay_qᶠᶜᶜ, Ay_qᶜᶠᶠ, Ay_qᶜᶠᶜ, Ay_qᶜᶜᶠ, Ay_qᶜᶜᶜ
export Az_qᶠᶠᶠ, Az_qᶠᶠᶜ, Az_qᶠᶜᶠ, Az_qᶠᶜᶜ, Az_qᶜᶠᶠ, Az_qᶜᶠᶜ, Az_qᶜᶜᶠ, Az_qᶜᶜᶜ

# Non-orthogonal metric and Hodge operators
export Jᶜᶜᵃ, Jᶠᶜᵃ, Jᶜᶠᵃ, Jᶜᶜᶜ, Jᶠᶜᶜ, Jᶜᶠᶜ, Jᶜᶜᶠ, Jᶠᶜᶠ, Jᶜᶠᶠ
export g¹¹ᶜᶜᵃ, g¹²ᶜᶜᵃ, g²¹ᶜᶜᵃ, g²²ᶜᶜᵃ, g¹¹ᶠᶜᵃ, g¹²ᶠᶜᵃ, g²¹ᶜᶠᵃ, g²²ᶜᶠᵃ
export g¹¹ᶜᶜᶜ, g¹²ᶜᶜᶜ, g²¹ᶜᶜᶜ, g²²ᶜᶜᶜ, g¹¹ᶠᶜᶜ, g¹²ᶠᶜᶜ, g²¹ᶜᶠᶜ, g²²ᶜᶠᶜ
export G¹¹ᶜᶜᵃ, G¹²ᶜᶜᵃ, G²¹ᶜᶜᵃ, G²²ᶜᶜᵃ, G¹¹ᶠᶜᵃ, G¹²ᶠᶜᵃ, G²¹ᶜᶠᵃ, G²²ᶜᶠᵃ
export G¹¹ᶜᶜᶜ, G¹²ᶜᶜᶜ, G²¹ᶜᶜᶜ, G²²ᶜᶜᶜ, G¹¹ᶠᶜᶜ, G¹²ᶠᶜᶜ, G²¹ᶜᶠᶜ, G²²ᶜᶠᶜ
export covariant_to_contravariant_flux_uᶠᶜᶜ, covariant_to_contravariant_flux_vᶜᶠᶜ
export covariant_to_contravariant_velocity_uᶠᶜᶜ, covariant_to_contravariant_velocity_vᶜᶠᶜ
export covariant_to_volume_flux_uᶠᶜᶜ, covariant_to_volume_flux_vᶜᶠᶜ
export hodge_compatible_boundary_flux_uᶠᶜᶜ, hodge_compatible_boundary_flux_vᶜᶠᶜ
export hodge_compatible_volume_flux_div_xyᶜᶜᶜ
export hodge_compatible_pressure_correction_uᶠᶜᶜ, hodge_compatible_pressure_correction_vᶜᶠᶜ
export covariant_gradient_xᶠᶜᶜ, covariant_gradient_yᶜᶠᶜ
export covariant_kinetic_energyᶜᶜᶜ, covariant_vertical_circulationᶠᶠᶜ, covariant_vertical_vorticityᶠᶠᶜ
export covariant_vertical_vorticity_componentᶠᶠᶜ
export covariant_rotational_advection_uᶠᶜᶜ, covariant_rotational_advection_vᶜᶠᶜ
export covariant_bernoulli_head_uᶠᶜᶜ, covariant_bernoulli_head_vᶜᶠᶜ

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
export divᶜᶜᶜ, div_xyᶜᶜᶜ, div_xyᶜᶜᶠ, horizontal_volume_flux_div_xyᶜᶜᶜ, ζ₃ᶠᶠᶜ
export ∇²ᶜᶜᶜ, ∇²ᶠᶜᶜ, ∇²ᶜᶠᶜ, ∇²ᶜᶜᶠ, ∇²hᶜᶜᶜ, ∇²hᶠᶜᶜ, ∇²hᶜᶠᶜ

# Interpolations
export ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
export ℑxyᶜᶜᵃ, ℑxyᶠᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶜᶠᵃ, ℑxzᶜᵃᶜ, ℑxzᶠᵃᶜ, ℑxzᶠᵃᶠ, ℑxzᶜᵃᶠ, ℑyzᵃᶜᶜ, ℑyzᵃᶠᶜ, ℑyzᵃᶠᶠ, ℑyzᵃᶜᶠ
export ℑxyzᶜᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶠ, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ
export active_weighted_ℑxyᶠᶜᶜ, active_weighted_ℑxyᶜᶠᶜ
export active_weighted_ℑxyᶜᶜᶜ, active_weighted_ℑxzᶜᶜᶜ, active_weighted_ℑyzᶜᶜᶜ

# Topology-aware operators
export δxTᶠᵃᵃ, δyTᵃᶠᵃ, δxTᶜᵃᵃ, δyTᵃᶜᵃ
export ∂xTᶠᶜᶠ, ∂yTᶜᶠᶠ
export ∂xᵣTᶠᶜᶠ, ∂yᵣTᶜᶠᶠ

# Reference frame conversion
export intrinsic_vector, extrinsic_vector

# Variable grid operators
export σⁿ, σ⁻, ∂t_σ

using Oceananigans.Grids: LatitudeLongitudeGrid, LLGOTF, XRegLLGOTF, YRegLLGOTF,
    EquiangularGnomonicCubedSpherePanel, OrthogonalSphericalShellGrid, SphericalShellGrid,
    OctaHEALPixMapping,
    RectilinearGrid, XRegularLLG, XRegularRG, YRegularLLG,
    YRegularRG, ZRegOrthogonalSphericalShellGrid, ZRegSphericalShellGrid, ZRegularLLG, ZRegularRG, AbstractGrid,
    Center, Face,
    octahealpix_covariant_xface_halo_source, octahealpix_covariant_yface_halo_source

#####
##### Convenient aliases
#####

const AG = AbstractGrid

const RG  = RectilinearGrid
const RGX = XRegularRG
const RGY = YRegularRG
const RGZ = ZRegularRG

const OSSG  = OrthogonalSphericalShellGrid
const OSSGZ = ZRegOrthogonalSphericalShellGrid
const SSG   = SphericalShellGrid
const SSGZ  = ZRegSphericalShellGrid
const SphericalShellLikeGrid = Union{OSSG, SSG}

const LLG  = LatitudeLongitudeGrid
const LLGX = XRegularLLG
const LLGY = YRegularLLG
const LLGZ = ZRegularLLG

# Vertically regular grids
const ZRG = Union{RGZ, OSSGZ, SSGZ, LLGZ}

const LLGF  = LLGOTF
const LLGFX = XRegLLGOTF
const LLGFY = YRegLLGOTF

const F = Face
const f = Face()
const C = Center
const c = Center()

include("difference_operators.jl")
include("interpolation_operators.jl")
include("interpolation_utils.jl")

include("reciprocal_metric_operators.jl")
include("spacings_and_areas_and_volumes.jl")
include("products_between_fields_and_grid_metrics.jl")
include("nonorthogonal_metric_operators.jl")

include("derivative_operators.jl")
include("constant_r_derivative_operators.jl")
include("divergence_operators.jl")
include("topology_aware_operators.jl")
include("vorticity_operators.jl")
include("laplacian_operators.jl")

include("time_variable_grid_operators.jl")
include("vector_rotation_operators.jl")

@inline xspacing(args...) = Δx(args...)
@inline yspacing(args...) = Δy(args...)
@inline zspacing(args...) = Δz(args...)
@inline λspacing(abs...)  = Δλ(abs...)
@inline φspacing(abs...)  = Δφ(abs...)
@inline rspacing(args...) = Δr(args...)
@inline xarea(args...)    = Ax(args...)
@inline yarea(args...)    = Ay(args...)
@inline zarea(args...)    = Az(args...)

# To be used as GridMetrics
struct XNode end
struct YNode end
struct ZNode end
struct ΛNode end
struct ΦNode end
struct RNode end

const x = XNode()
const y = YNode()
const z = ZNode()
const λ = ΛNode()
const φ = ΦNode()
const r = RNode()

end # module
