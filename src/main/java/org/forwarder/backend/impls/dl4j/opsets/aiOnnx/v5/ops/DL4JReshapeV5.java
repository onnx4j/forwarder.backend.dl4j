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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v5.ops;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JReshapeV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v5.ops.ReshapeV5;
import org.onnx4j.opsets.operator.OperatorOutputs;

import com.google.common.collect.Lists;

public class DL4JReshapeV5 extends DL4JReshapeV1 implements ReshapeV5 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		ReshapeInputsV5<INDArray> castedOperatorInputs = new ReshapeInputsV5<INDArray>(node, inputs);
		INDArray data = castedOperatorInputs.getData();
		INDArray shape = castedOperatorInputs.getShapeTensor();
		return new ReshapeOutputV5<INDArray>(this.reshape(data, shape));
	}

	protected INDArray reshape(INDArray data, INDArray shape) {
		return super.reshape(data, Lists.newArrayList(this.calcNewShape(data, shape)), null);
	}

	protected List<Long> calcNewShape(INDArray inputTensor, INDArray shapeTensor) {
		List<Long> newShape = new LinkedList<Long>();
		int autoExpandAxis = -1;
		for (int n = 0; n < shapeTensor.size(0); n++) {
			long len = shapeTensor.getLong(n);
			if (len > 0)
				newShape.add(n, len);
			else if (len == 0)
				newShape.add(n, inputTensor.shape()[n]);
			else if (len == -1) {
				if (autoExpandAxis != -1)
					throw new IllegalArgumentException(
							String.format("New shape can not hava more than one flag to execute auto-expand -> %s",
									Arrays.toString(inputTensor.shape())));

				autoExpandAxis = n;
				newShape.add(n, 1L);
			}
		}

		if (autoExpandAxis != -1) {
			//
			// do auto-expand
			//
			int sizeOfNewShape = 0;
			for (int n = 0; n < newShape.size(); n++) {
				if (sizeOfNewShape == 0)
					sizeOfNewShape += newShape.get(n);
				else
					sizeOfNewShape *= newShape.get(n);
			}
			int sizeOfData = 0;
			for (int n = 0; n < inputTensor.shape().length; n++) {
				if (sizeOfData == 0)
					sizeOfData += inputTensor.shape()[n];
				else
					sizeOfData *= inputTensor.shape()[n];
			}
			newShape.set(autoExpandAxis, (long) (sizeOfData / sizeOfNewShape));
		}

		return newShape;
	}

}