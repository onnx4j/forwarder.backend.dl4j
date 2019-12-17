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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.DL4JAiOnnxOperator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v1.ops.CastV1;

public class DL4JCastV1 extends DL4JAiOnnxOperator implements CastV1<INDArray> {

	protected static final Map<org.onnx4j.tensor.DataType, org.nd4j.linalg.api.buffer.DataType> DT_MAP = new HashMap<>();

	static {
		DT_MAP.put(org.onnx4j.tensor.DataType.UINT8, org.nd4j.linalg.api.buffer.DataType.UBYTE);
		DT_MAP.put(org.onnx4j.tensor.DataType.UINT16, org.nd4j.linalg.api.buffer.DataType.UINT16);
		DT_MAP.put(org.onnx4j.tensor.DataType.UINT32, org.nd4j.linalg.api.buffer.DataType.UINT32);
		DT_MAP.put(org.onnx4j.tensor.DataType.UINT64, org.nd4j.linalg.api.buffer.DataType.UINT64);
		DT_MAP.put(org.onnx4j.tensor.DataType.INT8, org.nd4j.linalg.api.buffer.DataType.BYTE);
		DT_MAP.put(org.onnx4j.tensor.DataType.INT16, org.nd4j.linalg.api.buffer.DataType.SHORT);
		DT_MAP.put(org.onnx4j.tensor.DataType.INT32, org.nd4j.linalg.api.buffer.DataType.INT);
		DT_MAP.put(org.onnx4j.tensor.DataType.INT64, org.nd4j.linalg.api.buffer.DataType.LONG);
		DT_MAP.put(org.onnx4j.tensor.DataType.FLOAT16, org.nd4j.linalg.api.buffer.DataType.HALF);
		DT_MAP.put(org.onnx4j.tensor.DataType.FLOAT, org.nd4j.linalg.api.buffer.DataType.FLOAT);
		DT_MAP.put(org.onnx4j.tensor.DataType.DOUBLE, org.nd4j.linalg.api.buffer.DataType.DOUBLE);
		DT_MAP.put(org.onnx4j.tensor.DataType.BOOL, org.nd4j.linalg.api.buffer.DataType.BOOL);
	}

	@Override
	public INDArray cast(INDArray t1, String to) {
		org.onnx4j.tensor.DataType onnx4jDataType = org.onnx4j.tensor.DataType.from(to);
		if (onnx4jDataType == null)
			throw new UnsupportedOperationException(String.format("Unsupported datatype named \"%s\" in ONNX4J", to));

		org.nd4j.linalg.api.buffer.DataType toDL4JDataType = DT_MAP.get(onnx4jDataType);
		if (toDL4JDataType == null)
			throw new UnsupportedOperationException(
					String.format("Can not convert datatype from \"%s\"(DL4J) to \"%s\"(ONN4J)", t1.dataType(), to));

		return this.cast(t1, toDL4JDataType);
	}

	protected INDArray cast(INDArray t1, org.nd4j.linalg.api.buffer.DataType to) {
		SameDiff sameDiff = DL4JSession.get();
		SDVariable t1Constant = sameDiff.constant(t1);
		t1Constant = sameDiff.castTo(t1Constant, to);
		return sameDiff.outputSingle(Collections.<String, INDArray>emptyMap(), t1Constant.getVarName());
	}

}