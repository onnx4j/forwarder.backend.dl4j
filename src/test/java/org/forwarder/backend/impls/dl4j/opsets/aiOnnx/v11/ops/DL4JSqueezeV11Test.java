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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v11.ops;

import java.util.Collections;
import java.util.List;

import org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v1.ops.DL4JSqueezeV1Test;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.primitives.Longs;

public class DL4JSqueezeV11Test extends DL4JSqueezeV1Test {

	@Test
	public void testWithNegativeAxes() throws Exception {
		this.testSqueeze(
				Nd4j.create(2, 1, 4), 
				Nd4j.create(2, 1, 4, 1), 
				Collections.unmodifiableList(Longs.asList(-1))
			);
		this.testSqueeze(
				Nd4j.create(2, 4), 
				Nd4j.create(2, 1, 4, 1), 
				Collections.unmodifiableList(Longs.asList(-1, -3))
			);
	}

	@Test
	public void testWithNegativeAxesButShapeEntryNotEqualToOne() throws Exception {
		thrown.expect(IllegalArgumentException.class);
		this.testSqueeze(
				null, 
				Nd4j.create(2, 1, 4, 1), 
				Collections.unmodifiableList(Longs.asList(-1, -2))
			);
	}

	@Override
	protected INDArray executeOperator(INDArray data, List<Long> axes) {
		DL4JSqueezeV11 operator = new DL4JSqueezeV11();
		INDArray y = operator.squeeze(data, axes);
		return y;
	}

}