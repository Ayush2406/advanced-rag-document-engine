from .vector_store import VectorStore
from ..config import settings
from typing import List,Any,Dict

class RAGRetriever:
    """ Handles query based retrieval from Vector DB"""
    def __init__(self,vector_store:VectorStore):
        self.vector_store=vector_store
    
    def retrieve(self,query:str,top_k:int = settings.TOP_K,score_threshold:float =0.0) ->List[Dict[str,Any]]:
        """
        Args:
            query: Query for vector DB
            top_k: number of top results to return
            score_threshold: minimum similarity threshold
        Returns:
            List of dict containing retrieved documents and metadata 
        """
        
        print(f"Retrieving documents for the query {query}")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")
        
        
        
        try:
            results = self.vector_store.similarity_search_with_score(query=query,k=top_k)
            
            if not results:
                print("No document found")
                return []
            
            #Processing the results
            retrieved_docs = []
            
            
                
            for i,(doc,score) in enumerate(results):
                distance = score
                similarity_score = 1-distance
                
                if similarity_score >= score_threshold:
                    retrieved_docs.append(
                        {'id':doc.id,
                        'content':doc.page_content,
                        'similarity_score':similarity_score,
                        'distance':distance,
                        'rank':i+1,
                        'metadata':doc.metadata}
                    )
            print(f"Retrieved {len(retrieved_docs)} documents after filtering.")
        
            
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval {e}")
            return []
        