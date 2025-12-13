import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection)
from azure.search.documents.indexes.models import (
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    AzureOpenAIEmbeddingSkill,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    CognitiveServicesAccountKey,
    SearchIndexerSkillset
)
from azure.search.documents.indexes.models import SearchIndexer
from openai import AzureOpenAI
from azure.search.documents import SearchClient # client to interact with the Search Index
from azure.search.documents.models import VectorizableTextQuery

load_dotenv()

AZURE_SEARCH_SERVICE: str = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_KEY: str = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ACCOUNT: str = os.getenv("AZURE_OPENAI_ACCOUNT")
AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY")
AZURE_AI_MULTISERVICE_ACCOUNT: str = os.getenv("AZURE_AI_MULTISERVICE_ACCOUNT")
AZURE_AI_MULTISERVICE_KEY: str = os.getenv("AZURE_AI_MULTISERVICE_KEY")
AZURE_STORAGE_CONNECTION: str = os.getenv("AZURE_STORAGE_CONNECTION")

credential = DefaultAzureCredential()

index_name = "sow-index"
index_client = SearchIndexClient(endpoint= AZURE_SEARCH_SERVICE, credential = credential)

field = [
    SearchField(name="id", type=SearchFieldDataType.String),
    SearchField(name="title", type=SearchFieldDataType.String),
    SearchField(name="chunk_id", type=SearchFieldDataType.String, sortable=True, filterable=True, key=True, facetable=True, analyzer_name="keyword"),
    SearchField(name="chunk", type=SearchFieldDataType.String, sortable=False, filterable=False, facetable=False),
    SearchField(
        name="text_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=3072,
        vector_search_profile_name="myHnswProfile"
    )
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="myHnsw"),
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                        vectorizer_name="myOpenAIVectorizer"
                    )
                ],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        vectorizer_name="myOpenAIVectorizer",
                        kind = "azureOpenAI",
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=AZURE_OPENAI_ACCOUNT,
                            deployment_name="text-embedding-3-large",
                            model_name="text-embedding-3-large",
                            api_key=AZURE_OPENAI_KEY
                        )
                    )
                ],
            )

# Semantic understanding rather than just keyword matching or vector similarity alone
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field = SemanticField(field_name="title"),
        content_fields = [SemanticField(field_name="chunk")]
    )
)

semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the index
index = SearchIndex(
    name=index_name,
    fields=field,
    vector_search=vector_search,
    semantic_search=semantic_search
)

result = index_client.create_or_update_index(index)
print("Index created:", result.name)

indexer_client = SearchIndexerClient(endpoint= AZURE_SEARCH_SERVICE, credential = credential)

container = SearchIndexerDataContainer(name="sow-container")
data_source_connection = SearchIndexerDataSourceConnection(
    name="sow-datasource",
    type="azureblob",
    connection_string=AZURE_STORAGE_CONNECTION,
    container=container
)

data_source = indexer_client.create_or_update_data_source_connection(data_source_connection)
print("Data source created:", data_source.name)

skillset_name = "statement-of-work-skillset"

split_skill = SplitSkill(
    name="split-skill",
    description="Splits skill to chunk documents",
    text_split_mode="pages",
    context="/document",
    inputs = [InputFieldMappingEntry(name="text", source="/document/content")],
    outputs = [OutputFieldMappingEntry(name="textItems", target_name="pages")], # Produces chunked text items called "textItems" and stores them in a field named "pages"
    maximum_page_length=2000, # maximum characters in each chunk/page
    page_overlap_length=500
)

embedding_skill = AzureOpenAIEmbeddingSkill(
    description="Generates embeddings for text chunks",
    name="azure-openai-embedding-skill",
    context="/document/pages/*", # apply to each chunk in the page
    resource_url=AZURE_OPENAI_ACCOUNT,
    deployment_name="text-embedding-3-large",
    model_name="text-embedding-3-large",
    api_key=AZURE_OPENAI_KEY,
    inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
    outputs=[OutputFieldMappingEntry(name="embedding", target_name="text_vector")], 
)

# map enriched data from skillset pipeline to the index

index_projections = SearchIndexerIndexProjection (
    selectors=[
        SearchIndexerIndexProjectionSelector(
            target_index_name=index_name, # which index to write to
            parent_key_field_name="id", # links each chunk to its parent/source document
            source_context="/document/pages/*",
            mappings=[ # how to map data to index fields
                InputFieldMappingEntry(name="chunk", source="/document/pages/*"),
                InputFieldMappingEntry(name="text_vector", source="/document/pages/*/text_vector"),
                InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),
            ],
        ),
    ],
    parameters=SearchIndexerIndexProjectionsParameters(
        projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS #only index the chunk, not the entire parent document
    )
)

cognitive_services_account = CognitiveServicesAccountKey(
    key=AZURE_AI_MULTISERVICE_KEY)
skills = [split_skill, embedding_skill]

skillset = SearchIndexerSkillset(
    name=skillset_name,
    description="Skillset for processing statement of work documents",
    skills=skills,
    index_projection=index_projections,
    cognitive_services_account=cognitive_services_account)

client = SearchIndexerClient(endpoint= AZURE_SEARCH_SERVICE, credential = credential)
created_skillset = client.create_or_update_skillset(skillset)
print("Skillset created:", created_skillset.name)


indexer_name = "sow-indexer"
indexer_parameters = None

indexer = SearchIndexer(
    name=indexer_name,
    description="Indexer for statement of work documents",
    skillset_name=skillset_name,
    target_index_name=index_name,
    data_source_name=data_source.name,
    parameters=indexer_parameters
)

indexer_client = SearchIndexerClient(endpoint= AZURE_SEARCH_SERVICE, credential = credential)
created_indexer = indexer_client.create_or_update_indexer(indexer)
