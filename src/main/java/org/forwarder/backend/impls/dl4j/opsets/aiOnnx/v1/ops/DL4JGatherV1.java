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

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.GatherV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JGatherV1 extends DL4JAiOnnxOperator implements GatherV1 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		GatherInputsV1<INDArray> castedOperatorInputs = new GatherInputsV1<INDArray>(node, inputs);
		INDArray data = castedOperatorInputs.getData();
		INDArray indices = castedOperatorInputs.getIndices();
		Long axis = castedOperatorInputs.getAxis();
		return new GatherOutputV1<INDArray>(this.gather(data, indices, axis));
	}

	protected INDArray gather(INDArray data, INDArray indices, Long axis) {
		throw new UnsupportedOperationException();
		/*SameDiff sameDiff = DL4JSession.get();
		SDVariable sdData = sameDiff.constant(data);
		SDVariable[] gathers = new SDVariable[indices.rows()];
		for (int q = 0; q < gathers.length; q++) {
			INDArray indicesQ = indices.getRow(q, false);
			long indicesQLength = indicesQ.shape()[indicesQ.shape().length - 1];
			indicesQ = indicesQ.reshape(indicesQLength);
			gathers[q] = sameDiff.gather(sdData, sameDiff.constant(indicesQ), axis.intValue());
		}

		// if (gathers.length > 1) {
		SDVariable concat = sameDiff.stack(0, gathers);
		return sameDiff.outputSingle(Collections.<String, INDArray> emptyMap(), concat.getVarName());
		// } else {
		// return sameDiff.outputSingle(Collections.<String,
		// INDArray>emptyMap(), gathers[0].getVarName());
		// }
*/	}

}