/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11.ops;

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSqueezeV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.guava.primitives.Ints;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v11.ops.SqueezeV11;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JSqueezeV11 extends DL4JSqueezeV1 implements SqueezeV11 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		SqueezeInputsV11<INDArray> castedOperatorInputs = new SqueezeInputsV11<INDArray>(node, inputs);
		INDArray data = castedOperatorInputs.getData();
		List<Long> axes = castedOperatorInputs.getAxes();
		return new SqueezeOutputV11<INDArray>(this.squeeze(data, axes));
	}

	@Override
	public INDArray squeeze(INDArray data, List<Long> axes) {
		return this.squeeze(data, Ints.toArray(axes));
	}

	@Override
	protected INDArray squeeze(INDArray data, int[] axes) {
		for (int n = 0; n < axes.length; n++) {
			if (axes[n] < 0) {
				int adjustedAxis = data.rank() + axes[n];
				if (adjustedAxis < 0 && adjustedAxis >= data.rank()) {
					throw new IllegalArgumentException(String.format(
							"The adjusted axis = %s, it must be a positive value and less than the rank of data(%s).",
							adjustedAxis, data.rank()));
				} else {
					axes[n] = adjustedAxis;
				}
			}
		}

		return super.squeeze(data, axes);
	}

}