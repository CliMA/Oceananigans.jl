using OrderedCollections: OrderedDict

function ordered_dict_show(dict::OrderedDict, padchar)
    name = "OrderedDict"
    N = length(dict)

    if N === 0
        return "$name with no entries"
    elseif N == 1
        return string("$name with 1 entry:", '\n',
                      padchar, "   └── ", dict.keys[1], " => ", summary(dict.vals[1]))
    else
        return string(name, " with $N entries:\n",
                      Tuple(string(padchar, "   ├── $name => ", summary(dict[name])) for name in dict.keys[1:end-1])...,
                            string(padchar, "   └── ", dict.keys[end], " => ", summary(dict.vals[end])))
    end
end

