module CubedSpheres

export
    ConformalCubedSphereGrid,
    λnodes, φnodes

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

validate_vertical_velocity_boundary_conditions(w::CubedSphereField) =
    [validate_vertical_velocity_boundary_conditions(w_face) for w_face in faces(w)]

#####
##### Applying flux boundary conditions
#####

import Oceananigans.Models.HydrostaticFreeSurfaceModels: apply_flux_bcs!

apply_flux_bcs!(Gcⁿ::CubedSphereField, events, c::CubedSphereField, arch, barrier, clock, model_fields) = [
    apply_flux_bcs!(face(Gcⁿ, face_number), events, face(c, face_number), arch, barrier, clock, model_fields)
    for face_number in 1:length(Gcⁿ.data.faces)
]

#####
##### NaN checker for cubed sphere fields
#####

import Oceananigans.Diagnostics: error_if_nan_in_field

function error_if_nan_in_field(field::CubedSphereField, name, clock)
    for (face_number, face_field) in enumerate(faces(field))
        error_if_nan_in_field(face_field, string(name) * " (face $face_number)", clock)
    end
end

#####
##### CFL for cubed sphere fields
#####

import Oceananigans.Diagnostics: accurate_cell_advection_timescale

function accurate_cell_advection_timescale(grid::ConformalCubedSphereGrid, velocities)

    min_timescale_on_faces = []

    for (face_number, grid_face) in enumerate(grid.faces)
        velocities_face = maybe_replace_with_face(velocities, grid, face_number)
        min_timescale_on_face = accurate_cell_advection_timescale(grid_face, velocities_face)
        push!(min_timescale_on_faces, min_timescale_on_face)
    end

    return minimum(min_timescale_on_faces)
end

#####
##### Output writing for cubed sphere fields
#####

import Oceananigans.OutputWriters: fetch_output

fetch_output(field::CubedSphereField, model, field_slicer) =
    Tuple(fetch_output(face_field, model, field_slicer) for face_field in faces(field))

#####
##### StateChecker for each face is useful for debugging
#####

import Oceananigans.Diagnostics: state_check

function state_check(field::CubedSphereField, name, pad)
    Nf = length(field.faces)
    for (face_number, field_face) in enumerate(field.faces)
        face_str = " face $face_number"
        state_check(field_face, string(name) * face_str, pad + length(face_str))

        # Leave empty line between fields for easier visual inspection.
        face_number == Nf && @info ""
    end
end

end # module
