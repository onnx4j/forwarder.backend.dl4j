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

import java.util.Collections;
import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.DL4JTestCase;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.primitives.Longs;

public class DL4JRedeceMaxV1Test extends DL4JTestCase {

	@Rule
	public ExpectedException thrown = ExpectedException.none();

	@Test
	public void testWithKeepdims() throws Exception {
		this.testReduceMax(
				Nd4j.create(new float[][] {
					{ 4.5f, 5.7f }
				}), 
				Nd4j.create(new float[][] {
					{ 1.0f, 1.2f },
					{ 2.3f, 3.4f },
					{ 4.5f, 5.7f }
				}), 
				Collections.unmodifiableList(Longs.asList(0)),
				1L);
		this.testReduceMax(
				Nd4j.create(new float[][] {
					{ 1.2f },
					{ 3.4f },
					{ 5.7f }
				}), 
				Nd4j.create(new float[][] {
					{ 1.0f, 1.2f },
					{ 2.3f, 3.4f },
					{ 4.5f, 5.7f }
				}), 
				Collections.unmodifiableList(Longs.asList(1)),
				1L);
	}

	@Test
	public void testWithoutKeepdims() throws Exception {
		this.testReduceMax(
				Nd4j.create(new float[] {
					4.5f, 5.7f
				}), 
				Nd4j.create(new float[][] {
					{ 1.0f, 1.2f },
					{ 2.3f, 3.4f },
					{ 4.5f, 5.7f }
				}), 
				Collections.unmodifiableList(Longs.asList(0)),
				0L);
		this.testReduceMax(
				Nd4j.create(new float[] {
					1.2f, 3.4f, 5.7f
				}), 
				Nd4j.create(new float[][] {
					{ 1.0f, 1.2f },
					{ 2.3f, 3.4f },
					{ 4.5f, 5.7f }
				}), 
				Collections.unmodifiableList(Longs.asList(1)),
				0L);
	}

	protected void testReduceMax(INDArray excepted, INDArray data, List<Long> axes, Long keepdims) throws Exception {
		try (DL4JSession session = new DL4JSession(null, null)) {
			INDArray y = this.executeOperator(data, axes, keepdims);
			System.out.println(String.format("{Excepted: %s} - {Actual: %s}", excepted.shapeInfoToString(),
					y.shapeInfoToString()));
			assertTrue(y.equals(excepted));
		}
	}

	protected INDArray executeOperator(INDArray data, List<Long> axes, Long keepdims) {
		DL4JReduceMaxV1 operator = new DL4JReduceMaxV1();
		INDArray y = operator.reduceMax(data, axes, keepdims);
		return y;
	}

}