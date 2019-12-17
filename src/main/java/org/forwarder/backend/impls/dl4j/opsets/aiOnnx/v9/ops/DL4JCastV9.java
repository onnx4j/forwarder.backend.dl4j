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

import java.util.HashMap;
import java.util.Map;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v6.ops.DL4JCastV6;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.onnx4j.opsets.aiOnnx.v9.ops.CastV9;

public class DL4JCastV9 extends DL4JCastV6 implements CastV9<INDArray> {

	protected static final Map<org.onnx4j.tensor.DataType, org.nd4j.linalg.api.buffer.DataType> DT_MAP = new HashMap<>(
			DL4JCastV6.DT_MAP);

	static {
		DT_MAP.put(org.onnx4j.tensor.DataType.STRING, org.nd4j.linalg.api.buffer.DataType.UTF8);
	}

	@Override
	public INDArray cast(INDArray t1, Long to) {
		return super.cast(t1, to);
	}

	@Override
	protected INDArray cast(INDArray t1, org.nd4j.linalg.api.buffer.DataType to) {
		if (t1.dataType().equals(org.nd4j.linalg.api.buffer.DataType.UTF8) && to.isNumerical())
			throw new UnsupportedOperationException(
					String.format("Can not convert datatype from \"%s\"(DL4J) to \"%s\"(ONN4J)", t1.dataType(), to));

		return super.cast(t1, to);
	}

}