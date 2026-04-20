// src/processors/processor.cpp — VisionProcessor static factory stub.

#include "geniex-proc/processor.h"

namespace geniex {

std::unique_ptr<VisionProcessor> VisionProcessor::from_config(const std::string& /*model_dir*/) {
    throw std::runtime_error("VisionProcessor::from_config: not yet implemented");
}

} // namespace geniex
