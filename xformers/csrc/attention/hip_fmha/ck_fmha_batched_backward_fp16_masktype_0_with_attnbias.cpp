#include <ck.hpp>
#include "ck_fmha_batched_backward.h"

template struct batched_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true,
    true>;

template struct batched_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true,
    false>;
