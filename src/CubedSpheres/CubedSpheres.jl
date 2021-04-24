module CubedSpheres

export ConformalCubedSphereGrid, MultiRegionGrid, get_region, regions

# Utilities for calculations on grids consisting of multiple "regions"
include("multi_region_grids.jl")
include("multi_region_data.jl")
include("multi_region_fields.jl")
include("get_region.jl")
include("multi_region_set!.jl")

include("cubed_sphere_utils.jl")
include("conformal_cubed_sphere_grid.jl")
include("cubed_sphere_exchange_bcs.jl")

include("cubed_sphere_halo_filling.jl")
include("cubed_sphere_kernel_launching.jl")

#####
##### Applying flux boundary conditions
#####

import Oceananigans.Models.HydrostaticFreeSurfaceModels: apply_flux_bcs!

function apply_flux_bcs!(Gcⁿ::AbstractMultiRegionField, events, c::AbstractMultiRegionField, arch, barrier, clock, model_fields)

    for (i, regional_Gcⁿ) in enumerate(regions(Gcⁿ))
        apply_flux_bcs!(regional_Gcⁿ, events, get_region(c, i), arch, barrier, clock, get_region(model_fields, i))
    end

    return nothing
end

#####
##### NaN checker for cubed sphere fields
#####

import Oceananigans.Diagnostics: error_if_nan_in_field

function error_if_nan_in_field(field::AbstractMultiRegionField, name, clock)
    for (region_index, regional_field) in enumerate(regions(field))
        error_if_nan_in_field(regional_field, string(name) * " (face $region_index)", clock)
    end
end

#####
##### CFL for cubed sphere fields
#####

import Oceananigans.Diagnostics: accurate_cell_advection_timescale

accurate_cell_advection_timescale(grid::MultiRegionGrid, U) =
    minimum(accurate_cell_advection_timescale(get_region(grid, i), get_region(U, i)) for i in nregions(grid))

#####
##### Output writing for cubed sphere fields
#####

import Oceananigans.OutputWriters: fetch_output

fetch_output(field::AbstractMultiRegionField, model, field_slicer) =
    Tuple(fetch_output(regional_field, model, field_slicer) for regional_field in regions(field))

#####
##### StateChecker for each face is useful for debugging
#####

import Oceananigans.Diagnostics: state_check

function state_check(field::AbstractMultiRegionField, name, pad)
    regional_fields = regions(field)
    Nf = length(regional_fields)
    for (region_index, regional_field) in enumerate(regional_fields)
        face_str = " face $region_index"
        state_check(regional_field, string(name) * face_str, pad + length(face_str))

        # Leave empty line between fields for easier visual inspection.
        region_index == Nf && @info ""
    end
end

end # module
