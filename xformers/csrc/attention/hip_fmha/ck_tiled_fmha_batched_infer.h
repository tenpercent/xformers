/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tiled_fmha_batched_infer_dispatch.h"
#include "ck_tiled_fmha_batched_infer_splitkv_dispatch.h"
#include "ck_tiled_fmha_seqlen_q_switch.h"

template <bool v>
struct has_mask_t : ck_tile::bool_constant<v> {};

template <bool v>
struct has_bias_t : ck_tile::bool_constant<v> {};

template <bool v>
struct has_dropout_t : ck_tile::bool_constant<v> {};

template <ck_tile::index_t v>
struct max_head_dimension_t : ck_tile::integral_constant<ck_tile::index_t, v> {
};

template <ck_tile::index_t v>
struct max_query_seqlen_t : ck_tile::integral_constant<ck_tile::index_t, v> {};

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
