/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tiled_fmha_batched_infer_dispatch.h"
#include "ck_tiled_fmha_batched_infer_splitkv_dispatch.h"
#include "ck_tiled_fmha_dispatch_tags.h"
#include "ck_tiled_fmha_seqlen_q_switch.h"

template <
    typename ScalarType,
    typename HasMask,
    typename HasBias,
    typename HasDropout,
    typename MaxHeadDimension>
void run_batched_infer_mask_bias_dropout_dispatch(
    BatchedForwardParams& param,
    hipStream_t stream) {
  // currently split-kv implementation does not support dropout
  if constexpr (!HasDropout::value) {
#ifndef FMHA_FWD_SPLITKV_NOT_USED
    if (param.use_split_kv) {
      FMHA_FWD_SEQLEN_Q_SWITCH(param.M, kMaxSeqlenQ, [&] {
        batched_infer_splitkv_mask_bias_dropout_dispatch<
            ScalarType,
            HasMask,
            HasBias,
            MaxHeadDimension,
            max_query_seqlen_t<kMaxSeqlenQ>>::Run(param, stream);
      });
    } else
#endif
      batched_infer_mask_bias_dropout_dispatch<
          ScalarType,
          HasMask,
          HasBias,
          HasDropout,
          MaxHeadDimension>::Run(param, stream);
  } else {
    batched_infer_mask_bias_dropout_dispatch<
        ScalarType,
        HasMask,
        HasBias,
        HasDropout,
        MaxHeadDimension>::Run(param, stream);
  }
};
