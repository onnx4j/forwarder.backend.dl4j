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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops;

import java.util.Arrays;
import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.Squeeze;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.Ints;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SqueezeV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JSqueezeV1 extends DL4JAiOnnxOperator implements SqueezeV1 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		SqueezeInputsV1<INDArray> castedOperatorInputs = new SqueezeInputsV1<INDArray>(node, inputs);
		INDArray data = castedOperatorInputs.getData();
		List<Long> axes = castedOperatorInputs.getAxes();
		return new SqueezeOutputV1<INDArray>(this.squeeze(data, axes));
	}

	protected INDArray squeeze(INDArray data, List<Long> axes) {
		return this.squeeze(data, Ints.toArray(axes));
	}

	protected INDArray squeeze(INDArray data, int[] axes) {
		long[] shape = data.shape();
		if (axes.length == 0) {
			axes = new int[shape.length];

			//
			// If axes is not provided, all the single dimensions will be
			// removed
			// from the shape.
			//
			for (int n = 0; n < shape.length; n++) {
				if (shape[n] == 1) {
					axes[n] = n;
				}
			}
		} else {
			//
			// If an axis is selected with shape entry not equal to one, an
			// error is raised.
			//
			for (int axis : axes) {
				if (shape[axis] != 1) {
					throw new IllegalArgumentException(String.format(
							"An axis(%s) is selected with shape entry expects to equal to 1, but is equal to %s", axis,
							shape[axis]));
				}
			}
		}

		//
		// axes集合必须符合从小到大（升序）排列
		//
		Arrays.sort(axes);

		Squeeze squeezeOp = new Squeeze();
		squeezeOp.addInputArgument(data);
		squeezeOp.setDimensions(axes);
		return Nd4j.exec(squeezeOp)[0];
	}

}