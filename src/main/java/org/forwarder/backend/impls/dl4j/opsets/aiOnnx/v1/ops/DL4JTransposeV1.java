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

import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.TransposeV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

import com.google.common.primitives.Ints;

public class DL4JTransposeV1 extends DL4JAiOnnxOperator
		implements TransposeV1 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		TransposeInputsV1<INDArray> castedOperatorInputs = new TransposeInputsV1<INDArray>(node, inputs);
		INDArray data = castedOperatorInputs.getData();
		List<Long> perm = castedOperatorInputs.getPerm();
		return new TransposeOutputV1<INDArray>(this.transpose(data, perm));
	}

	protected INDArray transpose(INDArray data, List<Long> perm) {
		return data.permute(Ints.toArray(perm));
	}

}