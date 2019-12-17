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
package org.forwarder.backend.impls.dl4j;

import org.forwarder.Session;
import org.forwarder.executor.Executor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DL4JSession extends Session<INDArray> {

	public static DL4JSession getSession() {
		Session<?> sess = TL_SESSION.get();
		if (sess == null)
			throw new RuntimeException("No forwarder session binded in this thread");

		if (DL4JSession.class.isInstance(sess) == false)
			throw new java.lang.ClassCastException("Session class type not match");

		DL4JSession dl4jSess = DL4JSession.class.cast(sess);
		return dl4jSess;
	}

	public static SameDiff get() {
		DL4JSession dl4jSess = DL4JSession.getSession();
		return dl4jSess.sameDiff;
	}

	private SameDiff sameDiff;

	public DL4JSession(Executor<INDArray> executor, DL4JBackend backend) {
		super(executor, backend);

		this.sameDiff = this.createSameDiff();
	}

	/**
	 * 创建后端实现Session实例
	 * 
	 * @return
	 */
	protected SameDiff createSameDiff() {
		return SameDiff.create().enableDebugMode();
	}

}