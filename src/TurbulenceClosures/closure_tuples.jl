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

for kernel_func in (:∂ⱼ_τ₁ⱼ, :∂ⱼ_τ₂ⱼ, :∂ⱼ_τ₃ⱼ, :∇_dot_qᶜ, :ivd_upper_diagonal, :ivd_lower_diagonal, :ivd_diagonal)
    @eval begin
        @inline $kernel_func(i, j, k, grid::AbstractGrid, closures::Tuple{<:Any}, Ks, args...) =
                    $kernel_func(i, j, k, grid, closures[1], Ks[1], args...)

        @inline $kernel_func(i, j, k, grid::AbstractGrid, closures::Tuple{<:Any, <:Any}, Ks, args...) = (
                    $kernel_func(i, j, k, grid, closures[1], Ks[1], args...)
                  + $kernel_func(i, j, k, grid, closures[2], Ks[2], args...))

        @inline $kernel_func(i, j, k, grid::AbstractGrid, closures::Tuple{<:Any, <:Any, <:Any}, Ks, args...) = (
                    $kernel_func(i, j, k, grid, closures[1], Ks[1], args...)
                  + $kernel_func(i, j, k, grid, closures[2], Ks[2], args...) 
                  + $kernel_func(i, j, k, grid, closures[3], Ks[3], args...))

        @inline $kernel_func(i, j, k, grid::AbstractGrid, closures::Tuple, Ks, args...) = (
                    $kernel_func(i, j, k, grid, closures[1:2], Ks[1:2], args...)
                  + $kernel_func(i, j, k, grid, closures[3:end], Ks[3:end], args...))
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
##### Support for VerticallyImplicit
#####

for coeff in (:νᶜᶜᶜ, :νᶠᶠᶜ, :νᶠᶜᶠ, :νᶜᶠᶠ, :κᶜᶜᶠ, :κᶜᶠᶜ, :κᶠᶜᶜ)
    @eval begin
        @inline $coeff(i, j, k, grid, clock, ν::Tuple{C1})     where C1       = $coeff(i, j, k, grid, clock, ν[1])
        @inline $coeff(i, j, k, grid, clock, ν::Tuple{C1, C2}) where {C1, C2} = $coeff(i, j, k, grid, clock, ν[1])   + $coeff(i, j, k, grid, clock, ν[2])
        @inline $coeff(i, j, k, grid, clock, ν::Tuple)                        = $coeff(i, j, k, grid, clock, ν[1:2]) + $coeff(i, j, k, grid, clock, ν[3:end])
    end
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
