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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v9.ops;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JCastV6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v9.ops.CastV9;
import org.onnx4j.opsets.operator.OperatorOutputs;

public class DL4JCastV9 extends DL4JCastV6 implements CastV9 {

	@Override
	public OperatorOutputs<INDArray> forward(Node node, Inputs inputs) {
		CastInputV9<INDArray> castedOperatorInputs = new CastInputV9<INDArray>(node, inputs);
		INDArray input = castedOperatorInputs.getInput();
		Long to = castedOperatorInputs.getToDTNumber();
		return new CastOutputV6<INDArray>(super.cast(input, super.toOnnx4jDataType(to)));
	}

	@Override
	protected INDArray cast(INDArray t1, org.nd4j.linalg.api.buffer.DataType to) {
		if (t1.dataType().equals(org.nd4j.linalg.api.buffer.DataType.UTF8) && to.isNumerical())
			throw new UnsupportedOperationException(
					String.format("Can not convert datatype from \"%s\"(DL4J) to \"%s\"(ONN4J)", t1.dataType(), to));

		return super.cast(t1, to);
	}

}