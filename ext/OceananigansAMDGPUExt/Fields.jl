module Fields

using AMDGPU
using Oceananigans.Fields: FieldBroadcastStyle, Field
using Base.Broadcast: Broadcasted

# broadcasting_abstract_fields.jl

Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::AMDGPU.ROCArrayStyle{N}) where N = FieldBroadcastStyle()

@inline function Base.Broadcast.materialize!(dest::Field, bc::Broadcasted{<:AMDGPU.ROCArrayStyle})
    if any(a isa OffsetArray for a in bc.args)
        return Base.Broadcast.materialize!(dest.data, bc)
    else
        return Base.Broadcast.materialize!(interior(dest), bc)
    end
end

end # module
