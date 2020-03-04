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

import static org.junit.Assert.assertTrue;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.DL4JTestCase;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JShapeV1Test extends DL4JTestCase {

	@Rule
	public ExpectedException thrown = ExpectedException.none();

	@Test
	public void testWithInt64Input() throws Exception {
		long[] shape = new long[] { 1, 2, 3 };
		
		this.testShape(
				Nd4j.create(
					shape, 
					new long[] { shape.length }, 
					DataType.LONG
				), 
				Nd4j.create(shape)
			);
	}

	protected void testShape(INDArray excepted, INDArray data) throws Exception {
		try (DL4JSession session = new DL4JSession(null)) {
			INDArray y = this.executeOperator(data);
			System.out.println(String.format("{Excepted: %s} - {Actual: %s}", excepted.shapeInfoToString(),
					y.shapeInfoToString()));
			assertTrue(y.equals(excepted));
		}
	}

	protected INDArray executeOperator(INDArray data) {
		DL4JShapeV1 operator = new DL4JShapeV1();
		INDArray y = operator.shape(data);
		return y;
	}

}