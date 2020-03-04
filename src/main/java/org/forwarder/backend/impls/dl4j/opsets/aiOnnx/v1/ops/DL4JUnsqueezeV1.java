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
import org.nd4j.linalg.factory.Nd4j;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.UnsqueezeV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

import com.google.common.primitives.Ints;

public class DL4JUnsqueezeV1 extends DL4JAiOnnxOperator implements UnsqueezeV1 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		UnsqueezeInputsV1<INDArray> castedOperatorInputs = new UnsqueezeInputsV1<INDArray>(node, inputs);
		INDArray data = castedOperatorInputs.getData();
		List<Long> axes = castedOperatorInputs.getAxes();
		return new UnsqueezeOutputV1<INDArray>(this.unsqueeze(data, axes));
	}

	protected INDArray unsqueeze(INDArray data, List<Long> axes) {
		return this.unsqueeze(data, Ints.toArray(axes));
	}

	protected INDArray unsqueeze(INDArray data, int[] axes) {
		//
		// axes集合必须符合从小到大（升序）排列
		//
		Arrays.sort(axes);

		int maxAxis = data.rank() + axes.length;

		INDArray intermediate = data;
		for (int axis : axes) {
			if (axis >= maxAxis)
				throw new IllegalArgumentException(
						String.format("The max axis value is %s, but the value of passed is %s.", maxAxis, axis));

			intermediate = Nd4j.expandDims(intermediate, axis);
		}
		return intermediate;
	}

}