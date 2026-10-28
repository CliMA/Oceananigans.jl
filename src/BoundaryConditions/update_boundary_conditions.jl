using Oceananigans: boundary_conditions

# `update_boundary_conditions!` refreshes conditions from the model state; the other three advance, cache and initialize the
# exterior states of open boundaries, which are prognostic and stepped by the time stepper.
for (fields_function, condition_function, arguments, sides) in
        ((:update_boundary_conditions!,     :update_boundary_condition!,     (:model,),                    (:(Val(:west)), :(Val(:east)), :(Val(:south)), :(Val(:north)), :(Val(:bottom)), :(Val(:top)), :(Val(:immersed)))),
         (:step_boundary_conditions!,       :step_boundary_condition!,       (:clock, :model_fields, :Δt), (:(West()), :(East()), :(South()), :(North()), :(Bottom()), :(Top()), :(Val(:immersed)))),
         (:cache_boundary_conditions!,      :cache_boundary_condition!,      (:clock, :model_fields),      (:(West()), :(East()), :(South()), :(North()), :(Bottom()), :(Top()), :(Val(:immersed)))),
         (:initialize_boundary_conditions!, :initialize_boundary_condition!, (:clock, :model_fields),      (:(West()), :(East()), :(South()), :(North()), :(Bottom()), :(Top()), :(Val(:immersed)))))

    west, east, south, north, bottom, top, immersed = sides

    @eval begin
        @inline $fields_function(bcs, field, $(arguments...)) = nothing
        @inline $condition_function(bc, side, field, $(arguments...)) = nothing

        function $fields_function(bcs::FieldBoundaryConditions, field, $(arguments...))
            $condition_function(bcs.west,     $west,     field, $(arguments...))
            $condition_function(bcs.east,     $east,     field, $(arguments...))
            $condition_function(bcs.south,    $south,    field, $(arguments...))
            $condition_function(bcs.north,    $north,    field, $(arguments...))
            $condition_function(bcs.bottom,   $bottom,   field, $(arguments...))
            $condition_function(bcs.top,      $top,      field, $(arguments...))
            $condition_function(bcs.immersed, $immersed, field, $(arguments...))
            return nothing
        end

        @inline $fields_function(fields::NamedTuple, $(arguments...)) = $fields_function(values(fields), $(arguments...))
        @inline $fields_function(::Tuple{}, $(arguments...)) = nothing

        @inline function $fields_function(fields::Tuple, $(arguments...))
            field = first(fields)
            $fields_function(boundary_conditions(field), field, $(arguments...))
            $fields_function(Base.tail(fields), $(arguments...))
            return nothing
        end
    end
end
