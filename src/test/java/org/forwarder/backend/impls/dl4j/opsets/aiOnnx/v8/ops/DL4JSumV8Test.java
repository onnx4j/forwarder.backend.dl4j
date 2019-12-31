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
package org.forwarder.backend.impls.dl4j.opsets.aiOnnx.v8.ops;

import static org.junit.Assert.assertTrue;

import java.util.LinkedList;
import java.util.List;

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.DL4JTestCase;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * All test cases can be found in <a href=
 * "https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules">numpy
 * broadcasting rules</a>
 * 
 * @author HarryLee
 *
 */
public class DL4JSumV8Test extends DL4JTestCase {

	@Rule
	public ExpectedException thrown = ExpectedException.none();
	@Test
	public void testWithSampleShape1() {
		try (INDArray excepted = Nd4j.create(256, 256, 3);
				INDArray a = Nd4j.create(256, 256, 3);
				INDArray b = Nd4j.create(256, 256, 3)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * Image  (3d array): 256 x 256 x 3
	 * Scale  (1d array):             3
	 * Result (3d array): 256 x 256 x 3
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting1() {
		try (INDArray excepted = Nd4j.create(256, 256, 3);
				INDArray a = Nd4j.create(256, 256, 3);
				INDArray b = Nd4j.create(3)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (4d array):  8 x 1 x 6 x 1
	 * B      (3d array):      7 x 1 x 5
	 * Result (4d array):  8 x 7 x 6 x 5
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting2() {
		try (INDArray excepted = Nd4j.create(8, 7, 6, 5);
				INDArray a = Nd4j.create(8, 1, 6, 1);
				INDArray b = Nd4j.create(7, 1, 5)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (2d array):  5 x 4
	 * B      (1d array):      1
	 * Result (2d array):  5 x 4
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting3() {
		try (INDArray excepted = Nd4j.create(5, 4); INDArray a = Nd4j.create(5, 4); INDArray b = Nd4j.create(1)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (2d array):  5 x 4
	 * B      (1d array):      4
	 * Result (2d array):  5 x 4
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting4() {
		try (INDArray excepted = Nd4j.create(5, 4); INDArray a = Nd4j.create(5, 4); INDArray b = Nd4j.create(4)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (3d array):  15 x 3 x 5
	 * B      (3d array):  15 x 1 x 5
	 * Result (3d array):  15 x 3 x 5
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting5() {
		try (INDArray excepted = Nd4j.create(15, 3, 5);
				INDArray a = Nd4j.create(15, 3, 5);
				INDArray b = Nd4j.create(15, 1, 5)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (3d array):  15 x 3 x 5
	 * B      (2d array):       3 x 5
	 * Result (3d array):  15 x 3 x 5
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting6() {
		try (INDArray excepted = Nd4j.create(15, 3, 5);
				INDArray a = Nd4j.create(15, 3, 5);
				INDArray b = Nd4j.create(3, 5)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (3d array):  15 x 3 x 5
	 * B      (2d array):       3 x 1
	 * Result (3d array):  15 x 3 x 5
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting7() {
		try (INDArray excepted = Nd4j.create(15, 3, 5);
				INDArray a = Nd4j.create(15, 3, 5);
				INDArray b = Nd4j.create(3, 1)) {
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (1d array):  3
	 * B      (1d array):  4 # trailing dimensions do not match
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting8() {
		try (INDArray excepted = null; INDArray a = Nd4j.create(3); INDArray b = Nd4j.create(4)) {
			thrown.expect(IllegalStateException.class);
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	/**
	 * <pre>
	 * A      (2d array):      2 x 1
	 * B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched
	 * </pre>
	 */
	@Test
	public void testWithBroadcasting9() {
		try (INDArray excepted = null; INDArray a = Nd4j.create(2, 1); INDArray b = Nd4j.create(8, 4, 3)) {
			thrown.expect(IllegalStateException.class);
			List<INDArray> dataList = new LinkedList<INDArray>();
			dataList.add(a);
			dataList.add(b);
			this.testSum(excepted, dataList);
		}
	}

	private void testSum(INDArray excepted, List<INDArray> dataList) {
		try (DL4JSession session = new DL4JSession(null, null)) {
			DL4JSumV8 operator = new DL4JSumV8();
			INDArray y = operator.sum(dataList);
			System.out.println(String.format("{Excepted: %s} - {Actual: %s}", excepted.shapeInfoToString(),
					y.shapeInfoToString()));
			assertTrue(y.equalShapes(excepted));
		}
	}

}