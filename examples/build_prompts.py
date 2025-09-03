from deepsight.core.query import QueryGenerator
from deepsight.core import ArtifactsManager
from deepsight.integrations.mlflow import MLflowManager
from deepsight.core.artifacts import ArtifactPaths
from deepsight.core.query.intelligence import IntelligenceClient, ProviderType, Providers

mlflow_mgr = MLflowManager(tracking_uri="http://localhost:5000", 
                            run_id="07c04cc42fd9461e98f7eb0bf42444fb", 
                            dwnd_dir="tmp"
                        )

artifacts = ArtifactsManager(
    sqlite_path="tmp/artifacts.db",
    artifacts_root="tmp/artifacts",
    mlflow_manager=mlflow_mgr,
)

# Register artifacts
#artifacts.register_artifact(run_id=mlflow_mgr.run_id, artifact_key=ArtifactPaths.DEEPCHECKS)
#artifacts.register_artifact(run_id=mlflow_mgr.run_id, artifact_key=ArtifactPaths.TRAINING)

deepchecks_artifact = artifacts.load_artifact(run_id=mlflow_mgr.run_id, 
                                    artifact_key=ArtifactPaths.DEEPCHECKS,
                                    download_if_missing=True
                                )


training_artifact = artifacts.load_artifact(run_id=mlflow_mgr.run_id, 
                                    artifact_key=ArtifactPaths.TRAINING,
                                    download_if_missing=True
                                )


query_generator = QueryGenerator()
prompt = query_generator.build_prompt([deepchecks_artifact,training_artifact])

print(prompt)

#intel = IntelligenceClient()
#response = intel.execute_query(prompt=prompt, provider_name=Providers.CURSOR)
#print(response.content)