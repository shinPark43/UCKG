# Embedding Process Docker Implementation Plan

## 1. Current System Analysis
- Entry point: `entry.py` handles data collection initialization
- Collections: CWE, CVE, D3FEND, ATT&CK, CAPEC
- Current flow: Waits for Neo4j → Initializes databases → Ends
- Note: No modifications needed to airflow/cron as embedding process will be automatically triggered through entry.py

## 2. Required Changes

### 2.1 Entry.py Modification
```python
# Add after all data source initializations
from embeddings.uckg_embedding_processor import UCKGEmbeddingProcessor

def run_embeddings():
    logger.info("Starting embedding generation for all nodes...")
    try:
        processor = UCKGEmbeddingProcessor()
        processor.process_all_types()
        logger.info("Embedding generation completed successfully")
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")

# Add after "All Data Sources Have Been Initialized!"
logger.info("Starting embedding process...")
run_embeddings()
```

### 2.2 Docker Configuration Updates

#### A. Update Dockerfile
```dockerfile
# Add to existing Dockerfile
# Install embedding dependencies
COPY graphrag/backend/embeddings/requirements.txt /app/embeddings/requirements.txt
RUN pip install -r /app/embeddings/requirements.txt

# Copy embedding code
COPY graphrag/backend/embeddings /app/embeddings
```

#### B. Environment Variables
Add to docker-compose.yml:
```yaml
environment:
  - NEO4J_URI=bolt://neo4j:7687
  - NEO4J_USER=neo4j
  - NEO4J_PASSWORD=abcd90909090
  - OLLAMA_URL=http://ollama:11434
  - EMBEDDING_MODEL=nomic-embed-text
```

## 3. Implementation Steps

### Step 1: Code Organization
- [x] Verify embedding code location: `graphrag/backend/embeddings/`
- [x] Ensure all dependencies are listed in requirements.txt
- [x] Check Neo4j and Ollama connection settings

### Step 2: Entry.py Integration
1. Import embedding processor
2. Add run_embeddings function
3. Call after data collection completion
4. Add proper logging and error handling

### Step 3: Docker Updates
1. Update Dockerfile with embedding dependencies
2. Add environment variables
3. Ensure proper volume mounts for data persistence
4. Update container dependencies

### Step 4: Testing Plan
1. Test data collection completion
2. Verify embedding process triggers
3. Check embedding results in Neo4j
4. Validate error handling
5. Test system recovery

## 4. Code Changes Required

### 4.1 entry.py
```python
# Add imports
import sys
sys.path.append("./graphrag/backend")
from embeddings.uckg_embedding_processor import UCKGEmbeddingProcessor

# Add after data collection
def run_embeddings():
    logger.info("Starting embedding generation...")
    processor = UCKGEmbeddingProcessor()
    processor.process_all_types()

# Add to main flow
if __name__ == "__main__":
    try:
        # ... existing data collection code ...
        
        logger.info("All Data Sources Have Been Initialized!")
        
        # Run embeddings
        logger.info("Starting embedding process...")
        run_embeddings()
        
        logger.info("Complete system initialization finished!")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        sys.exit(1)
```

### 4.2 docker-compose.yml
```yaml
services:
  uckg:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=abcd90909090
      - OLLAMA_URL=http://ollama:11434
      - EMBEDDING_MODEL=nomic-embed-text
    depends_on:
      - neo4j
      - ollama
```

## 5. Success Criteria
- [x] Data collection completes successfully
- [x] Embedding process starts automatically
- [x] All nodes receive embeddings
- [x] System handles errors gracefully
- [x] Logs provide clear status information

## 6. Monitoring
- Add log entries for:
  - Embedding process start/completion
  - Number of nodes processed
  - Error conditions
  - Processing time

## 7. Recovery Plan
- If embedding fails:
  1. Log error details
  2. Continue system operation
  3. Flag nodes for retry
  4. Alert via logs

## 8. Future Enhancements
1. Add progress tracking
2. Implement batch processing
3. Add retry mechanism
4. Performance optimization
5. Monitoring dashboard

## Commands to Test

```bash
# Build and run
docker-compose build
docker-compose up -d

# Check logs
docker-compose logs -f uckg

# Verify embeddings
docker-compose exec neo4j cypher-shell -u neo4j -p abcd90909090 \
  "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as nodes_with_embeddings"
```