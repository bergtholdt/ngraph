
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/codegen/compiler.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

TEST(DISABLED_include, complete)
{
    vector<string> include_files = {"ngraph/autodiff/adjoints.hpp",
                                    "ngraph/axis_set.hpp",
                                    "ngraph/axis_vector.hpp",
                                    "ngraph/builder/autobroadcast.hpp",
                                    "ngraph/builder/numpy_transpose.hpp",
                                    "ngraph/builder/reduce_ops.hpp",
                                    "ngraph/codegen/code_writer.hpp",
                                    "ngraph/codegen/compiler.hpp",
                                    "ngraph/codegen/execution_engine.hpp",
                                    "ngraph/coordinate.hpp",
                                    "ngraph/coordinate_diff.hpp",
                                    "ngraph/coordinate_transform.hpp",
                                    "ngraph/descriptor/buffer.hpp",
                                    "ngraph/descriptor/buffer_pos.hpp",
                                    "ngraph/descriptor/input.hpp",
                                    "ngraph/descriptor/layout/dense_tensor_view_layout.hpp",
                                    "ngraph/descriptor/layout/tensor_view_layout.hpp",
                                    "ngraph/descriptor/output.hpp",
                                    "ngraph/descriptor/tensor.hpp",
                                    "ngraph/descriptor/tensor_view.hpp",
                                    "ngraph/except.hpp",
                                    "ngraph/file_util.hpp",
                                    "ngraph/function.hpp",
                                    "ngraph/graph_util.hpp",
                                    "ngraph/log.hpp",
                                    "ngraph/ngraph.hpp",
                                    "ngraph/node.hpp",
                                    "ngraph/node_vector.hpp",
                                    "ngraph/op/abs.hpp",
                                    "ngraph/op/acos.hpp",
                                    "ngraph/op/add.hpp",
                                    "ngraph/op/allreduce.hpp",
                                    "ngraph/op/asin.hpp",
                                    "ngraph/op/atan.hpp",
                                    "ngraph/op/avg_pool.hpp",
                                    "ngraph/op/batch_norm.hpp",
                                    "ngraph/op/broadcast.hpp",
                                    "ngraph/op/ceiling.hpp",
                                    "ngraph/op/concat.hpp",
                                    "ngraph/op/constant.hpp",
                                    "ngraph/op/convert.hpp",
                                    "ngraph/op/convolution.hpp",
                                    "ngraph/op/cos.hpp",
                                    "ngraph/op/cosh.hpp",
                                    "ngraph/op/divide.hpp",
                                    "ngraph/op/dot.hpp",
                                    "ngraph/op/equal.hpp",
                                    "ngraph/op/exp.hpp",
                                    "ngraph/op/floor.hpp",
                                    "ngraph/op/function_call.hpp",
                                    "ngraph/op/get_output_element.hpp",
                                    "ngraph/op/greater.hpp",
                                    "ngraph/op/greater_eq.hpp",
                                    "ngraph/op/less.hpp",
                                    "ngraph/op/less_eq.hpp",
                                    "ngraph/op/log.hpp",
                                    "ngraph/op/max.hpp",
                                    "ngraph/op/maximum.hpp",
                                    "ngraph/op/max_pool.hpp",
                                    "ngraph/op/min.hpp",
                                    "ngraph/op/minimum.hpp",
                                    "ngraph/op/multiply.hpp",
                                    "ngraph/op/negative.hpp",
                                    "ngraph/op/not.hpp",
                                    "ngraph/op/not_equal.hpp",
                                    "ngraph/op/one_hot.hpp",
                                    "ngraph/op/op.hpp",
                                    "ngraph/op/pad.hpp",
                                    "ngraph/op/parameter.hpp",
                                    "ngraph/op/parameter_vector.hpp",
                                    "ngraph/op/power.hpp",
                                    "ngraph/op/product.hpp",
                                    "ngraph/op/reduce.hpp",
                                    "ngraph/op/reduce_window.hpp",
                                    "ngraph/op/relu.hpp",
                                    "ngraph/op/remainder.hpp",
                                    "ngraph/op/replace_slice.hpp",
                                    "ngraph/op/reshape.hpp",
                                    "ngraph/op/reverse.hpp",
                                    "ngraph/op/select.hpp",
                                    "ngraph/op/select_and_scatter.hpp",
                                    "ngraph/op/sign.hpp",
                                    "ngraph/op/sin.hpp",
                                    "ngraph/op/sinh.hpp",
                                    "ngraph/op/slice.hpp",
                                    "ngraph/op/sqrt.hpp",
                                    "ngraph/op/subtract.hpp",
                                    "ngraph/op/sum.hpp",
                                    "ngraph/op/tan.hpp",
                                    "ngraph/op/tanh.hpp",
                                    "ngraph/op/util/arithmetic_reduction.hpp",
                                    "ngraph/op/util/binary_elementwise.hpp",
                                    "ngraph/op/util/binary_elementwise_arithmetic.hpp",
                                    "ngraph/op/util/binary_elementwise_comparison.hpp",
                                    "ngraph/op/util/op_annotations.hpp",
                                    "ngraph/op/util/requires_tensor_view_args.hpp",
                                    "ngraph/op/util/unary_elementwise.hpp",
                                    "ngraph/op/util/unary_elementwise_arithmetic.hpp",
                                    "ngraph/pass/assign_layout.hpp",
                                    "ngraph/pass/assign_placement.hpp",
                                    "ngraph/pass/dump_sorted.hpp",
                                    "ngraph/pass/graph_rewrite.hpp",
                                    "ngraph/pass/inliner.hpp",
                                    "ngraph/pass/liveness.hpp",
                                    "ngraph/pass/manager.hpp",
                                    "ngraph/pass/manager_state.hpp",
                                    "ngraph/pass/memory_layout.hpp",
                                    "ngraph/pass/memory_visualize.hpp",
                                    "ngraph/pass/pass.hpp",
                                    "ngraph/pass/reshape_elimination.hpp",
                                    "ngraph/pass/visualize_tree.hpp",
                                    "ngraph/pattern/core_fusion.hpp",
                                    "ngraph/pattern/matcher.hpp",
                                    "ngraph/pattern/op/skip.hpp",
                                    "ngraph/pattern/op/label.hpp",
                                    "ngraph/pattern/op/pattern.hpp",
                                    "ngraph/placement.hpp",
                                    "ngraph/runtime/aligned_buffer.hpp",
                                    "ngraph/runtime/backend.hpp",
                                    "ngraph/runtime/cpu/cpu_backend.hpp",
                                    "ngraph/runtime/cpu/cpu_call_frame.hpp",
                                    "ngraph/runtime/cpu/cpu_eigen_utils.hpp",
                                    "ngraph/runtime/cpu/cpu_emitter.hpp",
                                    "ngraph/runtime/cpu/cpu_external_function.hpp",
                                    "ngraph/runtime/cpu/cpu_kernels.hpp",
                                    "ngraph/runtime/cpu/cpu_kernel_emitters.hpp",
                                    "ngraph/runtime/cpu/cpu_kernel_utils.hpp",
                                    "ngraph/runtime/cpu/cpu_layout_descriptor.hpp",
                                    "ngraph/runtime/cpu/cpu_manager.hpp",
                                    "ngraph/runtime/cpu/cpu_op_annotations.hpp",
                                    "ngraph/runtime/cpu/cpu_runtime_context.hpp",
                                    "ngraph/runtime/cpu/cpu_tensor_view.hpp",
                                    "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp",
                                    "ngraph/runtime/cpu/cpu_tracing.hpp",
                                    "ngraph/runtime/cpu/mkldnn_emitter.hpp",
                                    "ngraph/runtime/cpu/mkldnn_invoke.hpp",
                                    "ngraph/runtime/cpu/mkldnn_utils.hpp",
                                    "ngraph/runtime/cpu/op/convert_layout.hpp",
                                    "ngraph/runtime/cpu/op/matmul_bias.hpp",
                                    "ngraph/runtime/cpu/pass/cpu_assignment.hpp",
                                    "ngraph/runtime/cpu/pass/cpu_fusion.hpp",
                                    "ngraph/runtime/cpu/pass/cpu_layout.hpp",
                                    "ngraph/runtime/external_function.hpp",
                                    // "ngraph/runtime/gpu/gpu_backend.hpp",
                                    // "ngraph/runtime/gpu/gpu_call_frame.hpp",
                                    // "ngraph/runtime/gpu/gpu_cuda_context_manager.hpp",
                                    // "ngraph/runtime/gpu/gpu_cuda_function_builder.hpp",
                                    // "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp",
                                    // "ngraph/runtime/gpu/gpu_cuda_kernel_builder.hpp",
                                    // "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp",
                                    // "ngraph/runtime/gpu/gpu_emitter.hpp",
                                    // "ngraph/runtime/gpu/gpu_external_function.hpp",
                                    // "ngraph/runtime/gpu/gpu_kernel_emitters.hpp",
                                    // "ngraph/runtime/gpu/gpu_manager.hpp",
                                    // "ngraph/runtime/gpu/gpu_tensor_view.hpp",
                                    // "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp",
                                    // "ngraph/runtime/gpu/gpu_util.hpp",
                                    "ngraph/runtime/host_tensor_view.hpp",
                                    "ngraph/runtime/interpreter/int_backend.hpp",
                                    "ngraph/runtime/interpreter/int_call_frame.hpp",
                                    "ngraph/runtime/interpreter/int_external_function.hpp",
                                    "ngraph/runtime/interpreter/int_manager.hpp",
                                    "ngraph/runtime/reference/abs.hpp",
                                    "ngraph/runtime/reference/acos.hpp",
                                    "ngraph/runtime/reference/add.hpp",
                                    "ngraph/runtime/reference/allreduce.hpp",
                                    "ngraph/runtime/reference/asin.hpp",
                                    "ngraph/runtime/reference/atan.hpp",
                                    "ngraph/runtime/reference/avg_pool.hpp",
                                    "ngraph/runtime/reference/broadcast.hpp",
                                    "ngraph/runtime/reference/ceiling.hpp",
                                    "ngraph/runtime/reference/concat.hpp",
                                    "ngraph/runtime/reference/constant.hpp",
                                    "ngraph/runtime/reference/convert.hpp",
                                    "ngraph/runtime/reference/convolution.hpp",
                                    "ngraph/runtime/reference/copy.hpp",
                                    "ngraph/runtime/reference/cos.hpp",
                                    "ngraph/runtime/reference/cosh.hpp",
                                    "ngraph/runtime/reference/divide.hpp",
                                    "ngraph/runtime/reference/dot.hpp",
                                    "ngraph/runtime/reference/equal.hpp",
                                    "ngraph/runtime/reference/exp.hpp",
                                    "ngraph/runtime/reference/floor.hpp",
                                    "ngraph/runtime/reference/greater.hpp",
                                    "ngraph/runtime/reference/greater_eq.hpp",
                                    "ngraph/runtime/reference/less.hpp",
                                    "ngraph/runtime/reference/less_eq.hpp",
                                    "ngraph/runtime/reference/log.hpp",
                                    "ngraph/runtime/reference/max.hpp",
                                    "ngraph/runtime/reference/maximum.hpp",
                                    "ngraph/runtime/reference/max_pool.hpp",
                                    "ngraph/runtime/reference/min.hpp",
                                    "ngraph/runtime/reference/minimum.hpp",
                                    "ngraph/runtime/reference/multiply.hpp",
                                    "ngraph/runtime/reference/negate.hpp",
                                    "ngraph/runtime/reference/not.hpp",
                                    "ngraph/runtime/reference/not_equal.hpp",
                                    "ngraph/runtime/reference/one_hot.hpp",
                                    "ngraph/runtime/reference/pad.hpp",
                                    "ngraph/runtime/reference/power.hpp",
                                    "ngraph/runtime/reference/product.hpp",
                                    "ngraph/runtime/reference/reduce.hpp",
                                    "ngraph/runtime/reference/reduce_window.hpp",
                                    "ngraph/runtime/reference/relu.hpp",
                                    "ngraph/runtime/reference/replace_slice.hpp",
                                    "ngraph/runtime/reference/reshape.hpp",
                                    "ngraph/runtime/reference/reverse.hpp",
                                    "ngraph/runtime/reference/select.hpp",
                                    "ngraph/runtime/reference/select_and_scatter.hpp",
                                    "ngraph/runtime/reference/sign.hpp",
                                    "ngraph/runtime/reference/sin.hpp",
                                    "ngraph/runtime/reference/sinh.hpp",
                                    "ngraph/runtime/reference/slice.hpp",
                                    "ngraph/runtime/reference/sqrt.hpp",
                                    "ngraph/runtime/reference/subtract.hpp",
                                    "ngraph/runtime/reference/sum.hpp",
                                    "ngraph/runtime/reference/tan.hpp",
                                    "ngraph/runtime/reference/tanh.hpp",
                                    "ngraph/runtime/manager.hpp",
                                    "ngraph/runtime/tensor_view.hpp",
                                    "ngraph/serializer.hpp",
                                    "ngraph/shape.hpp",
                                    "ngraph/strides.hpp",
                                    "ngraph/type/element_type.hpp",
                                    "ngraph/type/type.hpp",
                                    "ngraph/util.hpp",
                                    "ngraph/uuid.hpp"};

    for (const string& include : include_files)
    {
        string source = "#include <" + include + ">\n ";

        stopwatch timer;
        timer.start();
        codegen::Compiler compiler;
        compiler.add_header_search_path(JSON_INCLUDES);
        auto module = compiler.compile(source);
        timer.stop();
        ASSERT_NE(nullptr, module) << source;
        // NGRAPH_INFO << timer.get_milliseconds() << " " << source;
    }
}
