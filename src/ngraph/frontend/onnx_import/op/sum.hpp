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

#pragma once

#include <numeric>

#include "ngraph/node_vector.hpp"
#include "ngraph/op/add.hpp"

#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            inline NodeVector sum(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};

                auto result =
                    std::accumulate(std::next(std::begin(ng_inputs)),
                                    std::end(ng_inputs),
                                    ng_inputs.front(),
                                    [](const std::shared_ptr<ngraph::Node>& arg0,
                                       const std::shared_ptr<ngraph::Node>& arg1) {
                                        return std::make_shared<ngraph::op::Add>(arg0, arg1);
                                    });

                return {result};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph