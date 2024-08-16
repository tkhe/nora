// Copyright (c) Facebook, Inc. and its affiliates.

#include <cuda_runtime_api.h>

namespace nora
{
    int get_cudart_version()
    {
        return CUDART_VERSION;
    }
}  // namespace nora
