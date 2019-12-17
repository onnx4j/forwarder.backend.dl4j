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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JCastV1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v6.ops.CastV6;

public class DL4JCastV6 extends DL4JCastV1 implements CastV6<INDArray> {

	@Override
	public INDArray cast(INDArray t1, Long to) {
		org.onnx4j.tensor.DataType onnx4jDataType = org.onnx4j.tensor.DataType.from(to.intValue());
		if (onnx4jDataType == null)
			throw new UnsupportedOperationException(String.format("Unsupported datatype id \"%s\" in ONNX4J", to));

		org.nd4j.linalg.api.buffer.DataType toDL4JDataType = DT_MAP.get(onnx4jDataType);
		if (toDL4JDataType == null)
			throw new UnsupportedOperationException(
					String.format("Can not convert datatype from \"%s\"(DL4J) to \"%s\"(ONN4J)", t1.dataType(), to));

		return super.cast(t1, toDL4JDataType);
	}

}