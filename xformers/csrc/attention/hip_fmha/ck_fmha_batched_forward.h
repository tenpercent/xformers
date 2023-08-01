#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_batched_mha_fwd_xdl_cshuffle_v1.hpp>
#include <ck/tensor_operation/gpu/device/tensor_specialization.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include "ck_fmha_util.h"

template <typename scalar_t, int32_t custom_mask_type>
void batched_forward_mask_type_dispatched(
    BatchedForwardParams& param,
    hipStream_t stream);

template <typename scalar_t>
void batched_forward(BatchedForwardParams& param, hipStream_t stream) {
  if (param.custom_mask_type == 0)
    batched_forward_mask_type_dispatched<scalar_t, 0>(param, stream);
  else if (param.custom_mask_type == 1)
    batched_forward_mask_type_dispatched<scalar_t, 1>(param, stream);
  else if (param.custom_mask_type == 2)
    batched_forward_mask_type_dispatched<scalar_t, 2>(param, stream);
  else
    throw std::runtime_error("Invalid custom_mask_type value");
};

template <typename scalar_t, int32_t custom_mask_type = 0>
void batched_forward_mask_type_dispatched(
    BatchedForwardParams& param,
    hipStream_t stream) {
  using PassThrough = ck::tensor_operation::element_wise::PassThrough;

  using GemmDataType = scalar_t;
  using ADataType = scalar_t;
  using B0DataType = scalar_t;
  using B1DataType = scalar_t;
  using AccDataType = F32;
  using CShuffleDataType = F32;
  using CDataType = scalar_t;
  using ZDataType = unsigned short;
  using LSEDataType = F32;
  using Acc0BiasDataType = ck::Tuple<>;
  using Acc1BiasDataType = ck::Tuple<>;

  static constexpr ck::index_t NumDimG = 2;
  static constexpr ck::index_t NumDimM = 1;
  static constexpr ck::index_t NumDimN = 1;
  static constexpr ck::index_t NumDimK = 1;
  static constexpr ck::index_t NumDimO = 1;

  using AElementOp = PassThrough;
  using B0ElementOp = PassThrough;
  using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
  using B1ElementOp = PassThrough;
  using CElementOp = PassThrough;

  static constexpr auto GemmSpec =
      ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
  static constexpr auto MaskingSpec =
      static_cast<ck::tensor_operation::device::MaskingSpecialization>(
          custom_mask_type);

  static constexpr auto TensorSpecA =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecB0 =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecB1 =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr auto TensorSpecC =
      ck::tensor_operation::device::TensorSpecialization::Default;
  static constexpr bool Deterministic = false;

  using DeviceOpInstance = ck::tensor_operation::device::
      DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle_V1<
          NumDimG,
          NumDimM,
          NumDimN,
          NumDimK,
          NumDimO,
          ADataType,
          B0DataType,
          B1DataType,
          CDataType,
          GemmDataType,
          ZDataType,
          LSEDataType,
          Acc0BiasDataType,
          Acc1BiasDataType,
          AccDataType,
          CShuffleDataType,
          AElementOp,
          B0ElementOp,
          Acc0ElementOp,
          B1ElementOp,
          CElementOp,
          GemmSpec,
          TensorSpecA,
          TensorSpecB0,
          TensorSpecB1,
          TensorSpecC,
          1,
          256,
          128, // MPerBlock
          128, // NPerBlock
          32, // KPerBlock
          32, // Gemm1NPerBlock
          32, // Gemm1KPerBlock
          8, // AK1
          8, // BK1
          2, // B1K1
          32, // MPerXDL
          32, // NPerXDL
          1, // MXdlPerWave
          4, // NXdlPerWave
          1, // Gemm1NXdlPerWave
          S<4, 64, 1>, // ABlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          8,
          8,
          true,
          S<4, 64, 1>, // BBlockTransfer
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          8,
          8,
          true,
          S<16, 16, 1>, // B1BlockTransfer
          S<0, 2, 1>,
          S<0, 2, 1>,
          1,
          2,
          2,
          false,
          1, // CShuffleMXdlPerWavePerShuffle
          1, // CShuffleNXdlPerWavePerShuffle
          S<1,
            64,
            1,
            4>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
          8, // CShuffleBlockTransferScalarPerVector_NPerBlock
          MaskingSpec, // MaskingSpecialization
          Deterministic>;

  float p_dropout = 1 - param.dropout_prob;
  ZDataType p_dropout_in_16bits = ZDataType(std::floor(p_dropout * 65535.0));
  float rp_dropout = 1.0 / p_dropout;
  float alpha = 1.f / std::sqrt(param.K);

  std::vector<ck::index_t> a_gs_ms_ks_lengths{
      param.B, param.num_heads, param.M, param.K};
  std::vector<ck::index_t> a_gs_ms_ks_strides{
      param.q_strides[0],
      param.q_strides[2],
      param.q_strides[1],
      param.q_strides[3]};

  std::vector<ck::index_t> b0_gs_ns_ks_lengths{
      param.B, param.num_heads, param.N, param.K};
  std::vector<ck::index_t> b0_gs_ns_ks_strides{
      param.k_strides[0],
      param.k_strides[2],
      param.k_strides[1],
      param.k_strides[3]};

  // to be changed to b1_gs_ns_os_lengths
  std::vector<ck::index_t> b1_gs_os_ns_lengths{
      param.B, param.num_heads, param.N, param.Kv};
  std::vector<ck::index_t> b1_gs_os_ns_strides{
      param.v_strides[0],
      param.v_strides[2],
      param.v_strides[3],
      param.v_strides[1]};

  std::vector<ck::index_t> c_gs_ms_os_lengths{
      param.B, param.num_heads, param.M, param.Kv};
  std::vector<ck::index_t> c_gs_ms_os_strides{
      param.out_strides[0],
      param.out_strides[2],
      param.out_strides[1],
      param.out_strides[3]};

  std::vector<ck::index_t> z_gs_ms_ns_lengths{
      param.B, param.num_heads, param.M, param.N};
  std::vector<ck::index_t> z_gs_ms_ns_strides{
      param.randvals_strides[0],
      param.randvals_strides[1],
      param.randvals_strides[2],
      param.randvals_strides[3]};

  std::vector<ck::index_t> lse_gs_ms_lengths{param.B, param.num_heads, param.M};

  auto a_element_op = AElementOp{};
  auto b0_element_op = B0ElementOp{};
  auto acc0_element_op = Acc0ElementOp{alpha};
  auto b1_element_op = B1ElementOp{};
  auto c_element_op = CElementOp{};

  // TODO, how to initialize seed, offset
  const uint64_t seed = 1;
  const uint64_t offset = 0;

  auto op = DeviceOpInstance{};
  auto invoker = op.MakeInvoker();
  auto arg_ptr = op.MakeArgumentPointer(
      param.q_ptr,
      param.k_ptr,
      param.v_ptr,
      param.out_ptr,
      param.randvals_ptr,
      param.logsumexp_ptr,
      {}, // std::array<void*, 1> p_acc0_biases;
      {}, // std::array<void*, 1> p_acc1_biases;
      a_gs_ms_ks_lengths,
      a_gs_ms_ks_strides,
      b0_gs_ns_ks_lengths,
      b0_gs_ns_ks_strides,
      b1_gs_os_ns_lengths,
      b1_gs_os_ns_strides,
      c_gs_ms_os_lengths,
      c_gs_ms_os_strides,
      z_gs_ms_ns_lengths,
      z_gs_ms_ns_strides,
      lse_gs_ms_lengths,
      {}, // std::array<std::vector<ck::index_t>,
          // 1>{acc0_biases_gs_ms_ns_lengths},
      {}, // std::array<std::vector<ck::index_t>,
          // 1>{acc0_biases_gs_ms_ns_strides},
      {}, // std::array<std::vector<ck::index_t>,
          // 1>{acc1_biases_gs_ms_os_lengths},
      {}, // std::array<std::vector<ck::index_t>,
          // 1>{acc1_biases_gs_ms_os_strides},
      a_element_op,
      b0_element_op,
      acc0_element_op,
      b1_element_op,
      c_element_op,
      param.dropout_prob, // dropout ratio
      {seed, offset}); // dropout random seed and offset, offset should be at
                       // least the number of elements on a thread
  SimpleDeviceMem workspace(op.GetWorkSpaceSize(arg_ptr.get()));

  op.SetWorkSpacePointer(arg_ptr.get(), workspace.GetDeviceBuffer());

  if (!op.IsSupportedArgument(arg_ptr.get())) {
    std::ostringstream ostr;

    ostr << op.GetTypeString() << " does not support this problem";

    throw std::runtime_error(ostr.str());
  }

  invoker.Run(arg_ptr.get(), StreamConfig{stream, false});
};
