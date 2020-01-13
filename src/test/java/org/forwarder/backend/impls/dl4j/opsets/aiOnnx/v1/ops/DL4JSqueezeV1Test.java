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

import com.google.common.collect.Lists;
import com.google.common.primitives.Longs;

public class DL4JSqueezeV1Test extends DL4JTestCase {

	@Rule
	public ExpectedException thrown = ExpectedException.none();

	@Test
	public void testWithAxes() {
		this.testSqueeze(
				Nd4j.create(2, 4, 1), 
				Nd4j.create(2, 1, 4, 1), 
				Collections.unmodifiableList(Longs.asList(1))
			);
		this.testSqueeze(
				Nd4j.create(2, 4), 
				Nd4j.create(2, 1, 4, 1), 
				Collections.unmodifiableList(Longs.asList(1, 3))
			);
	}

	@Test
	public void testWithShapeEntryNotEqualToOne() {
		thrown.expect(IllegalArgumentException.class);
		this.testSqueeze(
				null, 
				Nd4j.create(2, 1, 4, 1), 
				Collections.unmodifiableList(Longs.asList(0))
			);
	}

	@Test
	public void testWithEmptyAxes() {
		this.testSqueeze(
				Nd4j.create(2, 4), 
				Nd4j.create(2, 1, 4, 1), 
				Collections.unmodifiableList(Lists.newLinkedList())
			);
		this.testSqueeze(
				Nd4j.create(3, 4), 
				Nd4j.create(1, 3, 1, 4, 1), 
				Collections.unmodifiableList(Lists.newLinkedList())
			);
		/*this.testSqueeze(
				Nd4j.create(0,0), 
				Nd4j.create(1, 1, 1, 1, 1), 
				Lists.newLinkedList()
			);*/
	}

	protected void testSqueeze(INDArray excepted, INDArray data, List<Long> axes) {
		try (DL4JSession session = new DL4JSession(null, null)) {
			INDArray y = this.executeOperator(data, axes);
			System.out.println(String.format("{Excepted: %s} - {Actual: %s}", excepted.shapeInfoToString(),
					y.shapeInfoToString()));
			assertTrue(y.equalShapes(excepted));
		}
	}

	protected INDArray executeOperator(INDArray data, List<Long> axes) {
		DL4JSqueezeV1 operator = new DL4JSqueezeV1();
		INDArray y = operator.squeeze(data, axes);
		return y;
	}

}