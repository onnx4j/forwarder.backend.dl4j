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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.DL4JTestCase;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JCastV1Test extends DL4JTestCase {

	@Test
	public void testCastFloatToInt() {
		try (DL4JSession session = new DL4JSession(null, null)) {
			INDArray x0 = Nd4j.create(new float[] { 1.0123f, -1.4321f });
			assertEquals(x0.dataType(), DataType.FLOAT);
			DL4JCastV1 operator = new DL4JCastV1();
			INDArray y0 = operator.cast(x0, org.onnx4j.onnx.prototypes.OnnxProto3.TensorProto.DataType.INT32 + "");
			assertEquals(y0.dataType(), DataType.INT);
			assertTrue(Nd4j.create(new int[] { 1, -1 }, new long[] { 2 }, DataType.INT).equals(y0));
			y0.close();
			x0.close();
		}
	}

	@Test
	public void testCastDoubleToFloat() {
		try (DL4JSession session = new DL4JSession(null, null)) {
			INDArray x0 = Nd4j.create(new double[] { Double.MAX_VALUE, Double.MIN_VALUE });
			assertEquals(x0.dataType(), DataType.DOUBLE);
			DL4JCastV1 operator = new DL4JCastV1();
			INDArray y0 = operator.cast(x0, org.onnx4j.onnx.prototypes.OnnxProto3.TensorProto.DataType.FLOAT + "");
			assertEquals(y0.dataType(), DataType.FLOAT);
			assertTrue(Nd4j.create(new float[] { Float.MAX_VALUE, Float.MIN_VALUE }).equals(y0));
			y0.close();
			x0.close();
		}
	}

}