// Copyright (c) Facebook, Inc. and its affiliates.

#include <torch/extension.h>

#include "box_iou_rotated/box_iou_rotated.h"
#include "cocoeval/cocoeval.h"
#include "ms_deform_attn/ms_deform_attn.h"

namespace nora
{
#ifdef WITH_CUDA
    extern int get_cudart_version();
#endif

    std::string get_cuda_version()
    {
#ifdef WITH_CUDA
        std::ostringstream oss;

        oss << "CUDA ";

        // copied from
        // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
        auto printCudaStyleVersion = [&](int v)
        {
            oss << (v / 1000) << "." << (v / 10 % 100);
            if (v % 10 != 0)
            {
                oss << "." << (v % 10);
            }
        };
        printCudaStyleVersion(get_cudart_version());
        return oss.str();
#else
        return std::string("not available");
#endif
    }

    bool has_cuda()
    {
#ifdef WITH_CUDA
        return true;
#else
        return false;
#endif
    }

    // similar to
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
    std::string get_compiler_version()
    {
        std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

        { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
        { ss << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__; }
#endif

#if defined(_MSC_VER)
        { ss << "MSVC " << _MSC_FULL_VER; }
#endif
        return ss.str();
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
        m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");
        m.def("has_cuda", &has_cuda, "has_cuda");

        m.def("COCOevalAccumulate", &COCOeval::Accumulate, "COCOeval::Accumulate");
        m.def("COCOevalEvaluateImages", &COCOeval::EvaluateImages, "COCOeval::EvaluateImages");
        pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation").def(pybind11::init<>());
        pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation").def(pybind11::init<uint64_t, double, double, bool, bool>());

        m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
        m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
    }

    TORCH_LIBRARY(nora, m)
    {
        m.def("box_iou_rotated", &box_iou_rotated);
    }
}  // namespace nora
