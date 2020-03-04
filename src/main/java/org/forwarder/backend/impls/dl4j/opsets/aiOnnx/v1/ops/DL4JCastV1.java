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
import org.forwarder.backend.impls.dl4j.utils.DL4JDataTypeHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.CastV1;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JCastV1 extends DL4JAiOnnxOperator implements CastV1 {

	protected static final org.nd4j.linalg.api.buffer.DataType[] UNEXCEPTED_TYPES = new org.nd4j.linalg.api.buffer.DataType[] {
			org.nd4j.linalg.api.buffer.DataType.UTF8 };

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		CastInputsV1<INDArray> castedOperatorInputs = new CastInputsV1<INDArray>(node, inputs);
		INDArray input = castedOperatorInputs.getInput();
		String to = castedOperatorInputs.getTo();
		return new CastOutputV1<INDArray>(this.cast(input, this.toOnnx4jDataType(to)));
	}
	
	protected INDArray cast(INDArray t1, org.onnx4j.tensor.DataType toOnnx4jDataType) {
		org.nd4j.linalg.api.buffer.DataType toDL4JDataType = DL4JDataTypeHelper.toDl4jDataType(toOnnx4jDataType);
		if (toDL4JDataType == null)
			throw new UnsupportedOperationException(
					String.format("Can not convert datatype from \"%s\"(DL4J) to \"%s\"(ONN4J)", t1.dataType(), toOnnx4jDataType.getCode()));
		else {
			DL4JDataTypeHelper.ensureNotContainUnexceptedType(toDL4JDataType, UNEXCEPTED_TYPES);
			return this.cast(t1, toDL4JDataType);
		}
	}
	
	protected org.onnx4j.tensor.DataType toOnnx4jDataType(String onnx4jDataTypeName) {
		org.onnx4j.tensor.DataType onnx4jDataType = org.onnx4j.tensor.DataType.from(onnx4jDataTypeName);
		if (onnx4jDataType == null)
			throw new UnsupportedOperationException(String.format("Unsupported datatype \"%s\" in ONNX4J", onnx4jDataTypeName));
		else
			return onnx4jDataType;
	}

	protected INDArray cast(INDArray t1, org.nd4j.linalg.api.buffer.DataType to) {
		return t1.castTo(to);
	}

}