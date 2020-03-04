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

import java.util.UUID;

import org.forwarder.Session;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JSession extends Session<INDArray> {
	
	public static final String ND4J_WORKSPACE_NAME_PREFIX = "forwarder-session-";

	public static DL4JSession getSession() {
		Session<?> sess = TL_SESSION.get();
		if (sess == null)
			throw new RuntimeException("No forwarder session binded in this thread");

		if (DL4JSession.class.isInstance(sess) == false)
			throw new java.lang.ClassCastException("Session class type not match");

		DL4JSession dl4jSess = DL4JSession.class.cast(sess);
		return dl4jSess;
	}

	private MemoryWorkspace workspace;

	public DL4JSession(DL4JBackend backend) {
		super(backend);
		
		this.workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(
				WorkspaceConfiguration
					.builder()
	                .initialSize(100*1024*1024)
	                .maxSize(1024*1024*1024)
	                .policyAllocation(AllocationPolicy.STRICT)
	                .policyLocation(LocationPolicy.RAM)
	                .build(),
                ND4J_WORKSPACE_NAME_PREFIX + UUID.randomUUID());
		this.workspace.enableDebug(backend.getModel().getConfig().isDebug());
	}
	
	public MemoryWorkspace getMemoryWorkspace() {
		return this.workspace;
	}
	
	@Override
	public void close() throws Exception {
		super.close();
		workspace.destroyWorkspace();
		workspace.close();
		workspace = null;
	}

}