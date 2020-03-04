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

import org.forwarder.backend.impls.dl4j.DL4JSession;
import org.forwarder.backend.impls.dl4j.DL4JTestCase;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JGatherV1Test extends DL4JTestCase {

	@Test
	public void test1() throws Exception {
		this.testGather(
			Nd4j.create(new float[][][] {
				{
					{ 1.0f, 1.2f },
					{ 2.3f, 3.4f }
				},
				{
					{ 2.3f, 3.4f },
					{ 4.5f, 5.7f }
				}
			}), 
			Nd4j.create(new float[][] {
				{ 1.0f, 1.2f },
				{ 2.3f, 3.4f },
				{ 4.5f, 5.7f }
			}),
			Nd4j.createFromArray(new int[][] {
				{ 0, 1 },
				{ 1, 2 }
			}), 
			0L
		);
	}

	@Test
	public void test2() throws Exception {
		Long axis = 1L;
		INDArray data = Nd4j.create(new float[][] {
			{ 1.0f, 1.2f, 1.9f },
			{ 2.3f, 3.4f, 3.9f },
			{ 4.5f, 5.7f, 5.9f }
		});
		INDArray indices = Nd4j.createFromArray(new long[][] {
			{ 0, 2 }
		});
		INDArray excepted = Nd4j.create(new float[][][] {
			{
				{ 1.0f, 1.9f },
				{ 2.3f, 3.9f },
				{ 4.5f, 5.9f }
			}
		});
		this.testGather(excepted, data, indices, axis);
	}

	@Test
	public void test3() throws Exception {
		INDArray data = Nd4j.create(new long[][] {
			{ 1 , 2 },
			{ 3 , 4 },
		});
		INDArray indices = Nd4j.create(new long[][] {
			{ 0, 0 },
		});
		INDArray excepted = Nd4j.create(new long[][] {
			{ 1, 1 },
			{ 4, 3 },
		});
		this.testGather(excepted, data, indices, 1L);
	}

	@Test
	public void test4() throws Exception {
		try (DL4JSession session = new DL4JSession(null)) {
			Long axis = 1L;
			INDArray data = Nd4j.create(new long[][] {
				{ 1, 2, 3 },
				{ 4, 5, 6 },
			});
			INDArray indices = Nd4j.create(new long[][] {
				{ 0, 1 },
				{ 2, 0 },
			});
			INDArray excepted = Nd4j.create(new long[][] {
				{ 1, 2 },
				{ 6, 4 },
			});

			DL4JGatherV1 operator = new DL4JGatherV1();
			INDArray y0 = operator.gather(data, indices, axis);
			assertEquals(excepted, y0);
			
			y0.close();
			data.close();
			indices.close();
			excepted.close();
		}
	}

	@Test
	public void test5() throws Exception {
		try (DL4JSession session = new DL4JSession(null)) {
			Long axis = 1L;
			int[][][] dataarr = new int[][][] {
				{
					{ 1,  2},
					{ 3,  4},
					{ 5,  6},
					{ 7,  8}
				},
		       {
					{ 9, 10},
					{11, 12},
			        {13, 14},
			        {15, 16}
			   },
		       {
				   {17, 18},
				   {19, 20},
				   {21, 22},
				   {23, 24}
		       }
			};
			INDArray data = Nd4j.create(dataarr);
			INDArray indices = Nd4j.create(new long[][] {
				{ 1, 3 },
			});
			INDArray excepted = Nd4j.create(new long[][] {
				{ 1, 2 },
				{ 6, 4 },
			});

			DL4JGatherV1 operator = new DL4JGatherV1();
			INDArray y0 = operator.gather(data, indices, axis);
			assertEquals(excepted, y0);
			
			y0.close();
			data.close();
			indices.close();
			excepted.close();
		}
	}
	
	private void testGather(INDArray excepted, INDArray data, INDArray indices, Long axis) throws Exception {
		try (DL4JSession session = new DL4JSession(null)) {
			DL4JGatherV1 operator = new DL4JGatherV1();
			INDArray y0 = operator.gather(data, indices, axis);
			assertEquals(excepted, y0);
			
			y0.close();
			data.close();
			indices.close();
			excepted.close();
		}
	}

}