module CubedSpheres

export ConformalCubedSphereGrid, face, faces, λnodes, φnodes

include("cubed_sphere_utils.jl")
include("conformal_cubed_sphere_grid.jl")
include("cubed_sphere_exchange_bcs.jl")
include("cubed_sphere_faces.jl")
include("cubed_sphere_set!.jl")
include("cubed_sphere_halo_filling.jl")
include("cubed_sphere_kernel_launching.jl")

#####
##### Validating cubed sphere stuff
#####

import Oceananigans.Fields: validate_field_data
import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_vertical_velocity_boundary_conditions

function validate_field_data(X, Y, Z, data, grid::ConformalCubedSphereGrid)

    for (face_data, face_grid) in zip(data.faces, grid.faces)
        validate_field_data(X, Y, Z, face_data, face_grid)
    end

    return nothing
end

validate_vertical_velocity_boundary_conditions(w::AbstractCubedSphereField) =
    [validate_vertical_velocity_boundary_conditions(w_face) for w_face in faces(w)]

#####
##### Applying flux boundary conditions
#####

import Oceananigans.Models.HydrostaticFreeSurfaceModels: apply_flux_bcs!

function apply_flux_bcs!(Gcⁿ::AbstractCubedSphereField, events, c::AbstractCubedSphereField, arch, barrier, clock, model_fields)

    for (face_index, Gcⁿ_face) in enumerate(faces(Gcⁿ))
        apply_flux_bcs!(Gcⁿ_face, events, get_face(c, face_index), arch, barrier,
                        clock, get_face(model_fields, face_index))
    end

    return nothing
end

#####
##### NaN checker for cubed sphere fields
#####

import Oceananigans.Diagnostics: error_if_nan_in_field

function error_if_nan_in_field(field::AbstractCubedSphereField, name, clock)
    for (face_index, face_field) in enumerate(faces(field))
        error_if_nan_in_field(face_field, string(name) * " (face $face_index)", clock)
    end
end

#####
##### CFL for cubed sphere fields
#####

import Oceananigans.Diagnostics: accurate_cell_advection_timescale

function accurate_cell_advection_timescale(grid::ConformalCubedSphereGrid, velocities)

    min_timescale_on_faces = []

    for (face_index, grid_face) in enumerate(grid.faces)
        velocities_face = get_face(velocities, face_index)
        min_timescale_on_face = accurate_cell_advection_timescale(grid_face, velocities_face)
        push!(min_timescale_on_faces, min_timescale_on_face)
    end

    return minimum(min_timescale_on_faces)
end

#####
##### Output writing for cubed sphere fields
#####

import Oceananigans.OutputWriters: fetch_output

fetch_output(field::AbstractCubedSphereField, model, field_slicer) =
    Tuple(fetch_output(face_field, model, field_slicer) for face_field in faces(field))

#####
##### StateChecker for each face is useful for debugging
#####

import Oceananigans.Diagnostics: state_check

function state_check(field::AbstractCubedSphereField, name, pad)
    face_fields = faces(field)
    Nf = length(face_fields)
    for (face_index, face_field) in enumerate(face_fields)
        face_str = " face $face_index"
        state_check(face_field, string(name) * face_str, pad + length(face_str))

        # Leave empty line between fields for easier visual inspection.
        face_index == Nf && @info ""
    end
end

end # module
