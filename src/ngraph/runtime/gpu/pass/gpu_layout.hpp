//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/pass/pass.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"

#define LAYOUT_DECL(op_type)                                                                       \
    layout<op_type>(ngraph::runtime::gpu::GPU_ExternalFunction * external_function,                \
                    std::shared_ptr<ngraph::Node> node)

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                using LayoutFunction =
                    std::function<void(GPU_ExternalFunction*, std::shared_ptr<ngraph::Node>)>;

                using LayoutOpMap = std::unordered_map<std::type_index, LayoutFunction>;

                class GPULayout : public ngraph::pass::CallGraphPass
                {
                public:
                    GPULayout(GPU_ExternalFunction* external_function)
                        : m_external_function(external_function)
                    {
                    }
                    virtual bool
                        run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes) override;

                    template <typename OP>
                    static void
                        layout(ngraph::runtime::gpu::GPU_ExternalFunction* external_function,
                               std::shared_ptr<ngraph::Node> node);

                private:
                    GPU_ExternalFunction* m_external_function;
                };
            }
        }
    }
}
