#####
##### 'Tupled closure' implementation: 1-tuple, 2-tuple, and then n-tuple by induction
#####

function closure_summary(closures::Tuple, padchar="│")
    Nclosures = length(closures)
    if Nclosures == 1
        return string("Tuple with 1 closure:", '\n',
                      "$padchar   └── $(dict.keys[1]) => $(typeof(dict.vals[1]).name)")
    else
        return string("Tuple with $Nclosures closures:", '\n',
         Tuple(string("$padchar   ├── ", summary(c), '\n') for c in closures[1:end-1])...,
                      "$padchar   └── ", summary(closures[end]))
    end
end

#####
##### Kernel functions
#####

funcs     = [:∂ⱼ_τ₁ⱼ, :∂ⱼ_τ₂ⱼ, :∂ⱼ_τ₃ⱼ, :∇_dot_qᶜ, :maybe_tupled_ivd_upper_diagonal, :maybe_tupled_ivd_lower_diagonal]
alt_funcs = [:∂ⱼ_τ₁ⱼ, :∂ⱼ_τ₂ⱼ, :∂ⱼ_τ₃ⱼ, :∇_dot_qᶜ, :ivd_upper_diagonal, :ivd_lower_diagonal]

for (f, alt_f) in zip(funcs, alt_funcs)
    @eval begin
        @inline $f(i, j, k, grid, closures::Tuple{<:Any}, Ks, args...) =
                    $alt_f(i, j, k, grid, closures[1], Ks[1], args...)

        @inline $f(i, j, k, grid, closures::Tuple{<:Any, <:Any}, Ks, args...) = (
                    $alt_f(i, j, k, grid, closures[1], Ks[1], args...)
                  + $alt_f(i, j, k, grid, closures[2], Ks[2], args...))

        @inline $f(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, Ks, args...) = (
                    $alt_f(i, j, k, grid, closures[1], Ks[1], args...)
                  + $alt_f(i, j, k, grid, closures[2], Ks[2], args...) 
                  + $alt_f(i, j, k, grid, closures[3], Ks[3], args...))

        @inline $f(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any}, Ks, args...) = (
                    $alt_f(i, j, k, grid, closures[1], Ks[1], args...)
                  + $alt_f(i, j, k, grid, closures[2], Ks[2], args...) 
                  + $alt_f(i, j, k, grid, closures[3], Ks[3], args...) 
                  + $alt_f(i, j, k, grid, closures[4], Ks[4], args...))

        @inline $f(i, j, k, grid, closures::Tuple, Ks, args...) = (
                    $alt_f(i, j, k, grid, closures[1], Ks[1], args...)
                  + $f(i, j, k, grid, closures[2:end], Ks[2:end], args...))
    end
end

#####
##### Utilities
#####

with_tracers(tracers, closure_tuple::Tuple) = Tuple(with_tracers(tracers, closure) for closure in closure_tuple)

function calculate_diffusivities!(diffusivity_fields_tuple, closure_tuple::Tuple, args...)
    for (α, closure) in enumerate(closure_tuple)
        @inbounds diffusivity_fields = diffusivity_fields_tuple[α]
        calculate_diffusivities!(diffusivity_fields, closure, args...)
    end
    return nothing
end

function add_closure_specific_boundary_conditions(closure_tuple::Tuple, bcs, args...)
    # So the last closure in the tuple has the say...
    for closure in closure_tuple
        bcs = add_closure_specific_boundary_conditions(closure, bcs, args...)
    end
    return bcs
end

#####
##### Compiler-inferrable time_discretization for tuples
#####

const ETD = ExplicitTimeDiscretization
const VITD = VerticallyImplicitTimeDiscretization

@inline combine_time_discretizations(disc) = disc
@inline combine_time_discretizations(::ETD, ::VITD)  = VerticallyImplicitTimeDiscretization()
@inline combine_time_discretizations(::VITD, ::ETD)  = VerticallyImplicitTimeDiscretization()
@inline combine_time_discretizations(::VITD, ::VITD) = VerticallyImplicitTimeDiscretization()
@inline combine_time_discretizations(::ETD, ::ETD)   = ExplicitTimeDiscretization()

@inline combine_time_discretizations(d1, d2, other_discs...) =
    combine_time_discretizations(combine_time_discretizations(d1, d2), other_discs...)

@inline time_discretization(closures::Tuple) = combine_time_discretizations(time_discretization.(closures)...)

diffusivity_fields(grid, tracer_names, bcs, closure_tuple::Tuple) =
    Tuple(diffusivity_fields(grid, tracer_names, bcs, closure) for closure in closure_tuple)

