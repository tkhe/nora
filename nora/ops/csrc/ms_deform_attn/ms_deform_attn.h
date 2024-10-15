/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once

namespace nora
{
#ifdef WITH_CUDA
    at::Tensor ms_deform_attn_forward_cuda(
        const at::Tensor &value, 
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const int im2col_step
    );

    std::vector<at::Tensor> ms_deform_attn_backward_cuda(
        const at::Tensor &value, 
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const at::Tensor &grad_output,
        const int im2col_step
    );
#endif

    inline at::Tensor ms_deform_attn_forward(
        const at::Tensor &value, 
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const int im2col_step
    )
    {
        if (value.is_cuda())
        {
#ifdef WITH_CUDA
            return ms_deform_attn_forward_cuda(
                value,
                spatial_shapes,
                level_start_index,
                sampling_loc,
                attn_weight,
                im2col_step
            );
#else
            AT_ERROR("nora is not compiled with GPU support!");
#endif
        }
        else
        {
            AT_ERROR("This operator is not implemented on CPU");
        }
    }

    inline std::vector<at::Tensor> ms_deform_attn_backward(
        const at::Tensor &value, 
        const at::Tensor &spatial_shapes,
        const at::Tensor &level_start_index,
        const at::Tensor &sampling_loc,
        const at::Tensor &attn_weight,
        const at::Tensor &grad_output,
        const int im2col_step
    )
    {
        if (value.is_cuda())
        {
#ifdef WITH_CUDA
            return ms_deform_attn_backward_cuda(
                value,
                spatial_shapes,
                level_start_index,
                sampling_loc,
                attn_weight,
                grad_output,
                im2col_step
            );
#else
            AT_ERROR("nora is not compiled with GPU support!");
#endif
        }
        else
        {
            AT_ERROR("This operator is not implemented on CPU");
        }
    }
}  // namespace nora
