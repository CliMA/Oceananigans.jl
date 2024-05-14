using BitInformation

struct BitRounder{K}
    keepbits :: K
end

# number of keepbits (mantissa bits) for each variable
default_bit_rounding(::Val{name}) where name = 23   # single precision default
default_bit_rounding(::Val{:u}) = 2
default_bit_rounding(::Val{:v}) = 2
default_bit_rounding(::Val{:w}) = 2
default_bit_rounding(::Val{:T}) = 7
default_bit_rounding(::Val{:S}) = 16                # 12 at the surface, 16 deep ocean
default_bit_rounding(::Val{:η}) = 6

function BitRounding(outputs = nothing;
                     user_rounding...)

    keepbits = Dict()

    # TODO:
    # Check that the dimensions of keepbits are 
    # compatible with outputs if user_rounding 
    # contains an abstract array (support functions?)

    for name in keys(outputs)
        if name ∈ keys(user_rounding)
            keepbits[name] = user_rounding[name]
        else
            keepbits[name] = default_bit_rounding(Val(name))
        end
    end

    return BitRounding(keepbits)
end

# Getindex to allow indexing a BitRounder as 
Base.getindex(bit_rounding::BitRounding, name::Symbol) = BitRounding(bit_rounding[name])

function round_data!(output_array, bit_rounder::BitRounder) 
    
    # The actual rounding...
    keepbits = bit_rounder.keepbits

    # TODO: make sure that the rounding happens 
    # as we expect (priority to the vertical direction!)
    round!(output_array, keepbits)
    
    return output_array
end

    