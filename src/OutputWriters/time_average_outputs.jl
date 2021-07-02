time_average_outputs(schedule, outputs, model, field_slicer) = schedule, outputs # fallback

"""
    time_average_outputs(schedule::AveragedTimeInterval, outputs, model, field_slicer)

Wrap each `output` in a `WindowedTimeAverage` on the time-averaged `schedule` and with `field_slicer`.

Returns the `TimeInterval` associated with `schedule` and a `NamedTuple` or `Dict` of the wrapped
outputs.
"""
function time_average_outputs(schedule::AveragedTimeInterval, outputs::Dict, model, field_slicer)
    averaged_outputs = Dict(name => WindowedTimeAverage(output, model; schedule=copy(schedule), field_slicer=field_slicer)
                            for (name, output) in outputs)

    return TimeInterval(schedule), averaged_outputs
end

function time_average_outputs(schedule::AveragedTimeInterval, outputs::NamedTuple, model, field_slicer)
    output_names = Tuple(keys(outputs))

    output_values = Tuple(WindowedTimeAverage(outputs[name], model; schedule=copy(schedule), field_slicer=field_slicer)
                          for name in output_names)

    averaged_outputs = NamedTuple{output_names}(output_values)

    return TimeInterval(schedule), averaged_outputs
end
