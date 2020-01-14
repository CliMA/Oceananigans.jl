using OrderedCollections: OrderedDict

function ordered_dict_show(dict, padchar)
    if length(dict) == 0
        return string(typeof(dict), " with no entries")
    elseif length(dict) == 1
        return string(typeof(dict), " with 1 entry:", '\n',
                      padchar, "   └── ", dict.keys[1], " => ", typeof(dict.vals[1]))
    else
        return string(typeof(dict), " with $(length(dict)) entries:", '\n',
                      Tuple(string(padchar,
                                   "   ├── ", name, " => ", typeof(dict[name]), '\n')
                            for name in dict.keys[1:end-1]
                           )...,
                           padchar, "   └── ", dict.keys[end], " => ", typeof(dict.vals[end])
                      )
    end
end
