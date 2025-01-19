**Multimodal Recommendation System Using Image and Text Analysis for E-Commerce**

Welcome to the Multimodal Recommendation System repository! This project combines image and text modalities to deliver robust product recommendations in the fashion e-commerce domain. The system leverages CLIP embeddings for both images and text, applies semantic segmentation, performs keyword extraction, and includes a reranking strategy to balance visual vs. textual cues.

**Overview**

Goal: Enhance user experience on fashion e-commerce platforms by allowing queries that involve both image uploads and text descriptions (e.g., “blue floral dress with sleeves”).

**Key Techniques:**

CLIP: Alignment of image and text in a single embedding space  
Segmentation: Isolating clothing items to reduce background noise  
Keyword Extraction & Cosine Similarity: Weighting user-provided keywords to refine search results  
Two-Stage Retrieval & Reranking: Initial top-k retrieval followed by text-focused reranking  

**Features**

Multimodal Queries: Accept an image, text, or both.  
Cosine Similarity Search: Quickly find the nearest items in embedding space.  
Easy Customization: Adjust weighting parameters for image vs. text ( can use meta-learner to get tbe optimised weights)  
Segmentation Integration: Use Mask R-CNN or another segmentation method to crop out irrelevant backgrounds.  
Keyword Matching: Further enhance your search with keyword-level similarity matching.  

**Dataset**

DeepFashion dataset:  
~44,000 high-quality images in JPG/PNG format  
Variety of fashion categories (tops, pants, dresses, etc.)  
Dataset Link: [[link ](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok)](deepfashion_dataset2)  
Additional Text Labels/Descriptions to simulate real user queries (e.g., color, pattern, style).  
Evaluation doc: [[link ](https://docs.google.com/forms/d/e/1FAIpQLSenbdz3Fg_p7ssT8ArugoBNaK9sFnyUaHR-yogiEnGCXBeSsQ/viewform?pli=1)](results) 
