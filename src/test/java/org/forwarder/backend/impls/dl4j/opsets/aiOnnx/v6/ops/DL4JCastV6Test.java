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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.DL4JTestCase;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JCastV6Test extends DL4JTestCase {

	@Test
	public void testCastDoubleToBool() throws Exception {
		try (DL4JSession session = new DL4JSession(null, null)) {
			INDArray x0 = Nd4j.create(new double[] { Double.MAX_VALUE, 0.0d });
			assertEquals(x0.dataType(), DataType.DOUBLE);
			DL4JCastV6 operator = new DL4JCastV6();
			INDArray y0 = operator.cast(x0, org.onnx4j.prototypes.OnnxProto3.TensorProto.DataType.BOOL + "");
			assertEquals(y0.dataType(), DataType.BOOL);
			assertTrue(Nd4j.create(new boolean[] { true, false }).equals(y0));
			y0.close();
			x0.close();
		}
	}

}