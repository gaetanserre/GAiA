#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class scale_layer : public layer
{
public:
    explicit scale_layer(const std::string& name)
        : layer(name)
    {
    }
protected:
    tensors apply_impl(const tensors& input) const override
    {
        return {scale_tensors(input)};
    }
};

} } // namespace fdeep, namespace internal
