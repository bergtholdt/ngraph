/******************************************************************************
 * Copyright 2018 Intel Corporation
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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/shape.hpp"

#include "core/attribute.hpp"
#include "core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace attribute
        {
            /// \brief Get shape of kernel (filter) in pixels.
            ///
            /// \param node The Node ptr representing Conv or Pool operation.
            /// \return The kernel Shape object representing its dimensions (height, width, depth).
            Shape get_kernel_shape(const Node& node);

            /// \brief  Get number of pixels to stride operation by in each direction.
            ///
            /// \param node The Node ptr representing Conv or Pool operation.
            /// \param kernel_shape The shape of the kernel which we retrieve strides for.
            /// \return The kernel Shape object representing its dimensions (height, width, depth).
            Strides get_strides(const Node& node, const Shape& kernel_shape);

            /// \brief  Get number of pixels to stride operation by in each direction.
            ///
            /// \param node The Node ptr representing Conv or Pool operation.
            /// \return The kernel Shape object representing its dimensions (height, width, depth).
            Strides get_strides(const Node& node);

            /// \brief Get number of pixels for filter dilation in each direction.
            ///
            /// \param node The Node ptr representing ONNX operation.
            /// \return The Strides object containing number of pixels for filter dilation
            ///         (height, width, depth).
            Strides get_dilations(const Node& node);

            /// \brief Get padding values for the operation described by an ONNX node.
            /// \details If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID
            ///          values are calculated. Otherwise values are taken from the `pads` attribute.
            ///
            ///          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
            ///
            /// \param node The Node ptr representing ONNX operation.
            /// \param kernel_shape The shape of the kernel which we retrieve pads for.
            ///
            /// \return A pair of (padding_above, padding_below), which elements contains number of
            ///         pixels to pad in respective dimensions (height, width, depth).
            std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node,
                                                               const Shape& kernel_shape);

            
             /// \brief Get padding values for the operation described by an ONNX node.
             /// \details If `auto_pad` attribute is specified as SAME_UPPER or SAME_LOWER, or VALID
             ///          values are calculated. Otherwise values are taken from the `pads` attribute.
             ///
             ///          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
             ///
             /// \param node The Node ptr representing ONNX operation.
             ///
             /// \return A pair of (padding_above, padding_below), which elements contains number of
             ///         pixels to pad in respective dimensions (height, width, depth).
             
            inline std::pair<CoordinateDiff, CoordinateDiff> get_pads(const Node& node)
            {
                return get_pads(node, get_kernel_shape(node));
            }
        } // namespace attribute

    } // namespace  onnx_import

} // namespace  ngraph
