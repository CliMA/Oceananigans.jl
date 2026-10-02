using Adapt: Adapt

"""
    StepValue{N}(step)

The field `N` of the value stored in the one-element array `step`. Read in kernels with `step_value`.
"""
struct StepValue{N, S}
    step :: S
end

StepValue{N}(step) where N = StepValue{N, typeof(step)}(step)

Adapt.adapt_structure(to, v::StepValue{N}) where N = StepValue{N}(Adapt.adapt(to, v.step))

@inline step_value(value) = value
@inline step_value(v::StepValue{N}) where N = @inbounds getproperty(v.step[1], N)

"""
    capture_launches

Whether, on CUDA GPUs, `launch_captured!` records its kernel launches in a CUDA graph that is replayed on
later calls, instead of launching the kernels one by one.
"""
const capture_launches = Ref(true)

"""
    launch_captured!(launches!, arch, step_values, arguments...)

Call `launches!(step_values, arguments...)`, where `step_values` is a `NamedTuple` of the values that
change from one call to the next and `arguments` are the device-converted arguments that do not.

On CUDA GPUs, the kernels launched by `launches!` are recorded in a CUDA graph the first time it is
called with `arguments`, and the graph is replayed on later calls with the same `arguments`. During the
capture each entry of `step_values` is passed as a `StepValue`, so the kernels must read it with
`step_value`, and `launches!` must only launch kernels.
"""
launch_captured!(launches!, arch, step_values, arguments...) = launches!(step_values, arguments...)
